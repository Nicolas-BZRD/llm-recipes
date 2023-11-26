from dataclasses import dataclass
 
@dataclass
class dataset:
    file: str = None
    train_split: str = "train"
    val_split: str = "validation"
    context: bool = False
    few_shot: int = 0