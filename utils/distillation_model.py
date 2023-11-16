import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

class DistilModel(nn.Module):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher

        self._studentLogits = None
        self._teacherLogits = None
        self._studentLabels = None
        self._teacherLabels = None

    def forward(self, input_ids_student, attention_mask_student, labels_student, input_ids_teacher, attention_mask_teacher, labels_teacher):
        with torch.no_grad():            
            self._teacherLogits = self.teacher(
                input_ids = input_ids_teacher,
                attention_mask = attention_mask_teacher,
                labels = labels_teacher
            ).logits

        self._studentLogits = self.student(
            input_ids = input_ids_student,
            attention_mask = attention_mask_student,
            labels = labels_student
        ).logits

        self._studentLabels = labels_student
        self._teacherLabels = labels_teacher

        return self._studentLogits, self._teacherLogits

    @property
    def loss():
        return 1