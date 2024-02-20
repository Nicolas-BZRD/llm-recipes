import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer

def preprocess_distillation_batch(batch):
    batch_dict = {"student_" + key: value for key, value in batch[0].items()}
    batch_dict.update({"teacher_" + key: value for key,
                      value in batch[1].items()})
    return batch_dict


class DistillationModel(nn.Module):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()

    def forward(self, student_input_ids, student_attention_mask, student_labels, teacher_input_ids, teacher_attention_mask, teacher_labels):
        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                labels=teacher_labels
            )

        student_output = self.student(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=student_labels
        )
        return student_output, teacher_output


class DistillationLoss(nn.Module):
    def __init__(self, crossentropy_weight=1, distillation_weight=1, student_temperature=1, teacher_temperature=1, skip_student_eos=False, skip_teacher_eos=False, ignore_index=-100, debug=False, debug_rank=0, tokenizer_student=None, tokenizer_teacher=None):
        super().__init__()
        self.crossentropy_weight = crossentropy_weight
        self.distillation_weight = distillation_weight
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.skip_student_eos = skip_student_eos
        self.skip_teacher_eos = skip_teacher_eos
        self.ignore_index = ignore_index
        self.debug_rank = debug_rank
        self.debug = debug

        if self.debug:
            print("Distillation loss parameters:")
            print(f"Crossentropy weight: {crossentropy_weight}")
            print(f"Distillation weight: {distillation_weight}")
            print(f"Student temperature: {student_temperature}")
            print(f"Teacher temperature: {teacher_temperature}")
            print(f"Skip student eos: {skip_student_eos}")
            print(f"Skip teacher eos: {skip_teacher_eos}")
            print(f"Ignore index: {ignore_index}")
            print(f"Debug: {debug}")
            print(f"Debug rank: {debug_rank}")

            self.student_tokenizer = AutoTokenizer.from_pretrained(tokenizer_student)
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(tokenizer_teacher)

    def forward(self, student_predictions, teacher_predictions, student_targets, teacher_targets, rank=0):
        student = student_predictions.logits
        teacher = teacher_predictions.logits

        # Get answer first token and answer size
        student_answer_index, student_answer_size = self.__get_start_and_size_answers(
            student_targets)
        teacher_answer_index, teacher_answer_size = self.__get_start_and_size_answers(
            teacher_targets)

        # Avoid eos token, if needed
        if self.skip_student_eos: student_answer_size = [size-1 for size in student_answer_size]
        if self.skip_teacher_eos: teacher_answer_size = [size-1 for size in teacher_answer_size]

        # Align answer first token, pad to right and compute softmax
        for i in range(student.size(0)):
            shift = student_answer_index[i]
            size = student_answer_size[i]
            end_shift = shift+size
            student[i] = torch.cat((
                torch.nn.functional.softmax(student[i, shift:end_shift, :]/self.student_temperature, dim=-1),
                torch.zeros_like(student[i, :(student.size(1)-size), :])), dim=0
            )
        for i in range(teacher.size(0)):
            shift = teacher_answer_index[i]
            size = teacher_answer_size[i]
            end_shift = shift+size
            teacher[i] = torch.cat((
                torch.nn.functional.softmax(teacher[i, shift:end_shift, :]/self.teacher_temperature, dim=-1),
                torch.zeros_like(teacher[i, :(teacher.size(1)-size), :])), dim=0
            )

        # Cut to max answer length
        mex_length = max(max(student_answer_size), max(teacher_answer_size))
        student = student[:, :mex_length, :]
        teacher = teacher[:, :mex_length, :]

        if self.debug and rank == self.debug_rank:
            print("\n\n----------------------------------")
            print("------- Label / Prediction -------")
            print("----------------------------------")
            student_labels = [row[row != -100] for row in student_targets]
            teacher_labels = [row[row != -100] for row in teacher_targets]
            print("------- Label shape -------")
            print(f"Student label shape: {student_answer_size[0]}")
            print(f"Teacher label shape: {teacher_answer_size[0]}")
            print("------- Student Label -> Prediction -------")
            print(self.student_tokenizer.batch_decode(student_labels[0]))
            print(self.student_tokenizer.batch_decode(torch.argmax(
                student[0][:student_answer_size[0]], dim=-1)))
            print("------- Teacher Label -> Prediction -------")
            print(self.teacher_tokenizer.batch_decode(teacher_labels[0]))
            print(self.teacher_tokenizer.batch_decode(torch.argmax(
                teacher[0][:teacher_answer_size[0]], dim=-1)))
            print("------- Prediction Teacher -> Student  -------")
            print(self.teacher_tokenizer.batch_decode(torch.argmax(
                teacher[0][:teacher_answer_size[0]], dim=-1)))
            print(self.student_tokenizer.batch_decode(torch.argmax(
                student[0][:student_answer_size[0]], dim=-1)))
            print("------- Shape -------")
            print(f"Student shape: {student.size()}")
            print(f"Teacher shape: {teacher.size()}\n")

        # Sort in descending order to align probabilities
        student = student.sort(dim=-1, descending=True).values
        teacher = teacher.sort(dim=-1, descending=True).values

        # Pad to get same vocabulary size
        diff_size = student.size(2) - teacher.size(2)
        if diff_size > 0:
            teacher = F.pad(teacher, (0, diff_size), value=0)
        elif diff_size < 0:
            student = F.pad(student, (0, abs(diff_size)), value=0)

        if self.debug and rank == self.debug_rank:
            print("--------------------------------------------")
            print("---- Post-treatment tensor architecture ----")
            print("--------------------------------------------")
            print("------- Shape -------")
            print(f"Student shape: {student.size()}")
            print(f"Teacher shape: {teacher.size()}")
            print(" ------- First token -------")
            print(f"Student first logits: {student[0][0][:5].tolist()}")
            print(f"Teacher first logits: {teacher[0][0][:5].tolist()}")
            print(f"Student last logits: {student[0][0][-5:].tolist()}")
            print(f"Teacher last logits: {teacher[0][0][-5:].tolist()}")
            print(" ------- Last token -------")
            print(f"Student first logits: {student[0][-1][:5].tolist()}")
            print(f"Teacher first logits: {teacher[0][-1][:5].tolist()}")
            print(f"Student last logits: {student[0][-1][-5:].tolist()}")
            print(f"Teacher last logits: {teacher[0][-1][-5:].tolist()}\n")

        # Cross entropy loss
        crossentropy_loss = self.crossentropy_weight * student_predictions.loss

        distillation_loss = torch.zeros(student.size(0), device=student.device)
        for i in range(student.size(0)):
            size = min(student_answer_size[i], teacher_answer_size[i])
            distillation_loss[i] = abs(student[i][:size] - teacher[i][:size]).sum(-1).mean(-1)
        distillation_loss = distillation_loss.mean()
        distillation_loss = self.distillation_weight * (distillation_loss)

        if self.debug and rank == self.debug_rank:
            print("--------------------------------------")
            print("---------------- Loss ----------------")
            print("--------------------------------------")
            print(f"Crossentropy loss: {crossentropy_loss}")
            print(f"Distillation loss: {distillation_loss}")
            print(f"Total loss: {crossentropy_loss + distillation_loss}")

        return crossentropy_loss + distillation_loss, crossentropy_loss, distillation_loss

    def __get_start_and_size_answers(self, answer_tensors):
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            is_value = answer.eq(self.ignore_index)
            answers_size.append(len(answer) - int(is_value.sum()))
            indices = is_value.nonzero(as_tuple=True)[0]
            if len(indices) == 0 or indices[0] != 0:
                answers_index.append(0)
            else:
                diff_indices = indices[1:] - indices[:-1]
                break_index = (diff_indices != 1).nonzero()
                length = (break_index[0].item() +
                          1) if len(break_index) > 0 else len(indices)
                answers_index.append(length-1)
        return answers_index, answers_size