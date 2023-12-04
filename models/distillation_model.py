import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
student_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
teacher_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


class DistilModel(nn.Module):
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


def distil_loss(student_output, teacher_output, student_labels, teacher_labels, rank, Alpha=1, Beta=1, student_eos_skip=True, teacher_eos_skip=False, mask_labels=-100, debug=False):
    student = student_output.logits
    teacher = teacher_output.logits

    # Align answer first token and pad to right
    student_answer_index, student_answer_size = __get_start_and_size_answers(
        student_labels, mask_labels)
    teacher_answer_index, teacher_answer_size = __get_start_and_size_answers(
        teacher_labels, mask_labels)
    
    if student_eos_skip: student_answer_size = [size-1 for size in student_answer_size]
    if teacher_eos_skip: teacher_answer_size = [size-1 for size in teacher_answer_size]

    for i in range(student.size(0)):
        shift = student_answer_index[i]
        size = student_answer_size[i]
        end_shift = shift+size
        student[i] = torch.cat((
            torch.nn.functional.softmax(student[i, shift:end_shift, :], dim=-1),
            torch.zeros_like(student[i, :(student.size(1)-size), :])), dim=0
        )

    for i in range(teacher.size(0)):
        shift = teacher_answer_index[i]
        size = teacher_answer_size[i]
        end_shift = shift+size
        teacher[i] = torch.cat((
            torch.nn.functional.softmax(teacher[i, shift:end_shift, :], dim=-1),
            torch.zeros_like(teacher[i, :(teacher.size(1)-size), :])), dim=0
        )
    
    mex_length = max(max(student_answer_size), max(teacher_answer_size))
    student = student[:, :mex_length, :]
    teacher = teacher[:, :mex_length, :]
    
    if rank == 0 and debug:
        print("\n\n----------------------------------")
        print("------- Label / Prediction -------")
        print("----------------------------------")
        student_labels = [row[row != -100] for row in student_labels]
        teacher_labels = [row[row != -100] for row in teacher_labels]
        print("------- Label shape -------")
        print(f"Student label shape: {student_answer_size[0]}")
        print(f"Teacher label shape: {teacher_answer_size[0]}")
        print("------- Student Label -> Prediction -------")
        print(student_tokenizer.batch_decode(student_labels[0]))
        print(student_tokenizer.batch_decode(torch.argmax(student[0][:student_answer_size[0]], dim=-1)))
        print("------- Teacher Label -> Prediction -------")
        print(teacher_tokenizer.batch_decode(teacher_labels[0]))
        print(teacher_tokenizer.batch_decode(torch.argmax(teacher[0][:teacher_answer_size[0]], dim=-1)))
        print("------- Prediction Teacher -> Student  -------")
        print(teacher_tokenizer.batch_decode(torch.argmax(teacher[0][:teacher_answer_size[0]], dim=-1)))
        print(student_tokenizer.batch_decode(torch.argmax(student[0][:student_answer_size[0]], dim=-1)))
        print("------- Shape -------")
        print(f"Student shape: {student.size()}")
        print(f"Teacher shape: {teacher.size()}\n")

    student = student.sort(dim=-1, descending=True).values
    teacher = teacher.sort(dim=-1, descending=True).values

    diff_size = student.size(2) - teacher.size(2)
    if diff_size > 0:
        teacher = F.pad(teacher, (0, diff_size), value=0)
    elif diff_size < 0:
        student = F.pad(student, (0, abs(diff_size)), value=0)

    if rank == 0 and debug:
        print("--------------------------------------------")
        print("---- Post-treatment tensor architecture ----")
        print("--------------------------------------------")
        print("------- Shape -------")
        print(f"Student shape: {student.size()}")
        print(f"Teacher shape: {teacher.size()}")
        print(" ------- First token -------")
        print(f"Student first logits: {student[0][0][:5]}")
        print(f"Teacher first logits: {teacher[0][0][:5]}")
        print(f"Student last logits: {student[0][0][-5:]}")
        print(f"Teacher last logits: {teacher[0][0][-5:]}")
        print(" ------- Last token -------")
        print(f"Student first logits: {student[0][-1][:5]}")
        print(f"Teacher first logits: {teacher[0][-1][:5]}")
        print(f"Student last logits: {student[0][-1][-5:]}")
        print(f"Teacher last logits: {teacher[0][-1][-5:]}\n")
    
    # Compute loss
    dist_loss = abs(student - teacher).sum(-1)
    mask = (dist_loss != 0) & ~((0.9999 <= dist_loss) & (dist_loss <= 1.0001))
    dist_loss = ((dist_loss*mask).sum(dim=-1)/mask.sum(dim=-1)).nan_to_num()
    mask = (dist_loss != 0)
    dist_loss = ((dist_loss*mask).sum(dim=-1)/mask.sum(dim=-1)).nan_to_num()
    cross_loss = student_output.loss
    loss = Alpha*cross_loss + Beta*dist_loss

    if rank == 0 and debug:
        print("------------------------------")
        print("------------ Loss ------------")
        print("------------------------------")
        tmp = abs(student - teacher).sum(-1)
        print(f"student-teacher: {tmp[0]}")
        tmp_mask = (tmp != 0) & ~((0.9999 <= tmp) & (tmp <= 1.0001))
        tmp = ((tmp*tmp_mask).sum(dim=-1)/tmp_mask.sum(dim=-1)).nan_to_num()
        print(f"Mask: {tmp_mask[0]}")
        print(f"Mean: {tmp[0]}")
        tmp_mask = (tmp != 0)
        tmp = ((tmp*tmp_mask).sum(dim=-1)/tmp_mask.sum(dim=-1)).nan_to_num()
        print(f"Mask: {tmp_mask[0]}")
        print(f"Mean: {tmp[0]}")
        print(f"Distil Loss all batch: {dist_loss}")
        print(f"Cross Loss all batch: {cross_loss}")
        print("------------------------------------------------------------------------------")

    return loss, cross_loss, dist_loss


def __get_start_and_size_answers(answer_tensors, mask=-100):
    answers_index = []
    answers_size = []

    for answer in answer_tensors:
        is_value = answer.eq(mask)
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


def preprocess_distillation_batch(batch):
    batch_dict = {"student_" + key: value for key, value in batch[0].items()}
    batch_dict.update({"teacher_" + key: value for key,
                      value in batch[1].items()})
    return batch_dict
