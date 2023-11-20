import torch
import dataclasses
import torch.optim as optim

from policies import AnyPrecisionAdamW
from policies import apply_fsdp_checkpointing
from models.fsdp import fsdp_auto_wrap_policy
from configs import fsdp_config as FSDP_CONFIG
from models.distillation_model import DistilModel
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs.configs_utils import generate_peft_config, update_config
from peft import get_peft_model, prepare_model_for_int8_training
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from models.tools import (
    freeze_transformer_layers,
    print_model_size,
    get_policies
)

def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def load_model(train_config, rank):
    use_cache = False if train_config.enable_fsdp else True
    def load():
        return AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else False,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
        )
    
    if not train_config.enable_fsdp:
        model = load()
        
    elif train_config.enable_fsdp:
        if train_config.low_cpu_fsdp:
            if rank == 0:
                model = load()
            else:
                model_config = AutoModelForCausalLM.from_pretrained(
                    train_config.model_name)
                model_config.use_cache = use_cache
                with torch.device("meta"):
                    model = AutoModelForCausalLM.from_config(model_config)
        else:
            model = load()

        if train_config.use_fast_kernels:
            """
            For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
            using of Flash Attention or Xformer memory-efficient kernels
            based on the hardware being used. This would speed up fine-tuning.
            """
            model = BetterTransformer.transform(model)
            
    print_model_size(model, train_config, rank)
    return model

def set_model(model, train_config, fsdp_config, rank, kwargs):
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif train_config.freeze_layers:
        freeze_transformer_layers(train_config.num_freeze_layers)

    if train_config.enable_fsdp:
        if fsdp_config.pure_bf16: model.to(torch.bfloat16)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer) #TODO: Modify to be general (parameter)

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )

        if fsdp_config.fsdp_activation_checkpointing: apply_fsdp_checkpointing(model)
        return model
    else:
        if train_config.quantization: return model
        else:
            return model.to(f"cuda:{rank}")

def get_model(train_config, fsdp_config, rank, kwargs):
    model = load_model(train_config, rank)
    model = set_model(model, train_config, fsdp_config, rank, kwargs)
    tokenizer = load_tokenizer(train_config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model

def get_distillation_models(train_config, distil_config, fsdp_config, rank, kwargs):
    student_tokenizer, student_model = get_model(train_config, fsdp_config, rank, kwargs)

    teacher_fsdp_config = FSDP_CONFIG()
    update_config((teacher_fsdp_config), **dataclasses.asdict(distil_config))
    print(teacher_fsdp_config)
    print(distil_config)
    exit()
    teacher_tokenizer, teacher_model = get_model(distil_config, distil_config, rank, kwargs)

    return student_tokenizer, teacher_tokenizer, DistilModel(student_model, teacher_model)

def get_optimizer(model, train_config, fsdp_config):
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        return AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        return optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )