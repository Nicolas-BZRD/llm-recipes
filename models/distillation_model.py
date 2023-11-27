import torch
import torch.nn as nn


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


def distil_loss(student_output, teacher_output, student_labels, teacher_labels, Alpha=1, Beta=1, mask_labels=-100):
    student = student_output.logits
    teacher = teacher_output.logits

    student_answer_index, student_answer_size = __get_start_and_size_answers(
        student_labels, mask_labels)
    teacher_answer_index, teacher_answer_size = __get_start_and_size_answers(
        teacher_labels, mask_labels)

    # Align answer first token and pad to right
    for i in range(student.size(0)):
        shift = student_answer_index[i]
        size = student_answer_size[i]
        end_shift = shift + size
        student[i] = torch.cat((student[i, shift:end_shift, :], torch.zeros_like(
            student[i, :(student.size(1)-size), :])), dim=0)
    student = student[:, :max(student_answer_size), :]

    for i in range(teacher.size(0)):
        shift = teacher_answer_index[i]
        size = teacher_answer_size[i]
        end_shift = shift+size
        teacher[i] = torch.cat((teacher[i, shift:end_shift, :], torch.zeros_like(
            teacher[i, :(teacher.size(1)-size), :])), dim=0)
    teacher = teacher[:, :max(teacher_answer_size), :]

    # Softmax predictions
    student = torch.nn.functional.softmax(
        student, dim=-1).sort(dim=-1, descending=True).values
    teacher = torch.nn.functional.softmax(
        teacher, dim=-1).sort(dim=-1, descending=True).values

    # Align same dictionary size
    min_size = min(teacher.size(-1), student.size(-1))
    student = torch.narrow(student, -1, 0, min_size)
    teacher = torch.narrow(teacher, -1, 0, min_size)

    # Pad to get same number of token per sentence
    diff_size = student.size(1) - teacher.size(1)
    if diff_size > 0:
        teacher = torch.cat((
            teacher,
            torch.zeros((teacher.size(0), diff_size,
                        teacher.size(2)), device=teacher.device)
        ), dim=1)

    elif diff_size < 0:
        student = torch.cat((
            student,
            torch.zeros((student.size(0), abs(diff_size),
                        student.size(2)), device=student.device)
        ), dim=1)

    # Compute loss
    cross_loss = Alpha * student_output.loss

    distil_loss = abs(student - teacher).sum(-1)
    distil_loss = torch.where(distil_loss > 0.98, torch.tensor(
        0.).to(distil_loss.device), distil_loss).mean().mean()
    distil_loss = Beta * distil_loss

    return (cross_loss+distil_loss), cross_loss, distil_loss


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
            answers_index.append(length)
    return answers_index, answers_size


def preprocess_distillation_batch(batch):
    batch_dict = {"student_" + key: value for key, value in batch[0].items()}
    batch_dict.update({"teacher_" + key: value for key,
                      value in batch[1].items()})
    return batch_dict
