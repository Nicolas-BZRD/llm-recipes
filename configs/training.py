from dataclasses import dataclass

@dataclass
class train_config:
    project_name: str=None
    model_name: str="meta-llama/Llama-2-7b-hf"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=8
    batching_strategy: str="padding"
    context_length: int=None
    gradient_accumulation_steps: int=1
    num_epochs: int=1
    num_workers_dataloader: int=2
    lr: float=1e-6
    weight_decay: float=0.1
    pct_start=0.1
    div_factor=2
    final_div_factor=5
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    peft_method: str = "lora"
    use_peft: bool=False
    output_dir: str = ""
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    save_model: bool = True
    save_step: int = 1000
    save_optimizer: bool=False
    use_fast_kernels: bool = False
    distillation: bool = False
    save_all: bool = False
    training_size: int = 1
    encoder_decoder: bool = False