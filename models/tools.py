import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist

from pkg_resources import packaging
from policies import fpSixteen, bfSixteen, get_wrapper

def get_parameter_dtypes(model):
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False

def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            print(
                f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")            

def get_policies(cfg, rank):
    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_wrapper()
    return mixed_precision_policy, wrapping_policy

def print_model_size(model, config, rank: int = 0) -> None:
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        print(
            f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")