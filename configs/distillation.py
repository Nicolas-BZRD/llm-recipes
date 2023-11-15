from dataclasses import dataclass

@dataclass
class distillation_config:
    teacher_model: str = "meta-llama/Llama-2-7b-hf"