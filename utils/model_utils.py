from pkg_resources import packaging
import dataclasses

import torch
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from policies import apply_fsdp_checkpointing

from utils import fsdp_auto_wrap_policy
from utils.config_utils import generate_peft_config, update_config
from configs import fsdp_config as FSDP_CONFIG
from utils.train_utils import (
    freeze_transformer_layers,
    print_model_size,
    get_policies
)
from utils.distillation_model import DistilModel


def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def fetch_model(train_config, rank):
    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception(
                "Pytorch >= 20230701 build is required to run with low_cpu_fsdp config")
        if rank == 0:
            model = AutoModelForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
        else:
            model_config = AutoModelForCausalLM.from_pretrained(
                train_config.model_name)
            model_config.use_cache = use_cache
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
        )
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print(
                "Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    print_model_size(model, train_config, rank)
    return model


def set_model(model, train_config, fsdp_config, rank, kwargs):
    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(
            fsdp_config, rank)
        # TODO Modify to be general
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(
            model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(
                offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(
                device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")
    return model


def prepare_model(train_config, fsdp_config, rank, kwargs):
    model = fetch_model(train_config, rank)
    model = set_model(model, train_config, fsdp_config, rank, kwargs)
    tokenizer = load_tokenizer(train_config.model_name)

    return tokenizer, model


def prepare_model_distillation(train_config, distil_config, fsdp_config, rank, kwargs):
    student_tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

    teacher_tokenizer = AutoTokenizer.from_pretrained(distil_config.model_name)
    teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    student_model = fetch_model(train_config, rank)
    student_model = set_model(
        student_model, train_config, fsdp_config, rank, kwargs)

    teacher_fsdp_config = FSDP_CONFIG()
    update_config((teacher_fsdp_config), **dataclasses.asdict(distil_config))
    teacher_model = fetch_model(distil_config, rank)
    teacher_model = set_model(
        teacher_model, distil_config, fsdp_config, rank, kwargs)

    return student_tokenizer, teacher_tokenizer, DistilModel(student_model, teacher_model)
