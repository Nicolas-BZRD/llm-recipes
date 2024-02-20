from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class distillation_config:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    quantization: bool = False
    use_fast_kernels: bool = False
    use_peft: bool = False
    freeze_layers: bool = False
    num_freeze_layers: int = 0
    cross_entropy_factor: float = 1
    distil_factor: float = 1.5
    student_temperature: float = 1
    teacher_temperature: float = 1
    encoder_decoder: bool = False
    
    # FSDP Config
    mixed_precision: bool = False
    use_fp16: bool = False
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    pure_bf16: bool = False
    optimizer: str = "AdamW"