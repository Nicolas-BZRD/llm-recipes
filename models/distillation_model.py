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
                input_ids = teacher_input_ids,
                attention_mask = teacher_attention_mask,
                labels = teacher_labels
            )

        student_output = self.student(
            input_ids = student_input_ids,
            attention_mask = student_attention_mask,
            labels = student_labels
        )
        return student_output, teacher_output

def distil_loss(student_output, teacher_output, student_labels, teacher_labels, Alpha=1, Beta=10, T=1, mask_labels=-100):
    student = torch.nn.functional.softmax(student_output.logits / T, dim=-1).sort(dim=-1, descending=True).values
    student_mask = (student_labels != mask_labels)
    student = student_mask.unsqueeze(2) * student

    teacher = torch.nn.functional.softmax(teacher_output.logits / T, dim=-1).sort(dim=-1, descending=True).values
    teacher_mask = (teacher_labels != mask_labels)
    teacher = teacher_mask.unsqueeze(2) * teacher

    for dim in [1,2]:
        min_size = min(teacher.size(dim), student.size(dim))
        student = torch.narrow(student, dim, 0, min_size)
        teacher = torch.narrow(teacher, dim, 0, min_size)
    
    cross_loss = Alpha * student_output.loss
    dist_loss = Beta * abs(student-teacher).sum(dim=-1).mean()**T
    return (cross_loss + dist_loss), cross_loss, dist_loss

def preprocess_distillation_batch(batch):
    batch_dict = {"student_" + key: value for key, value in batch[0].items()}
    batch_dict.update({"teacher_" + key: value for key,
                      value in batch[1].items()})
    return batch_dict