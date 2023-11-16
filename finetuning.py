import os

import fire
import random
import torch
import torch.optim as optim

from configs import fsdp_config as FSDP_CONFIG
from configs import train_config as TRAIN_CONFIG
from configs import distillation_config as DISTIL_CONFIG
from policies import AnyPrecisionAdamW

from utils.config_utils import (
    update_config,
    update_sub_config
)
from utils.dataset_utils import (
    get_dataloader,
    get_dataloader_distillation
)

from utils.train_utils import (
    train,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
)

from utils.model_utils import (
    prepare_model,
    prepare_model_distillation
)
from utils.config_utils import generate_dataset_config

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config, distil_config = TRAIN_CONFIG(), FSDP_CONFIG(), DISTIL_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    update_sub_config((distil_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
    else:
        rank = 0

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load Model and Tokenizer
    if not train_config.distillation:
        tokenizer, model = prepare_model(
            train_config, fsdp_config, rank, kwargs)
    else:
        student_tokenizer, teacher_tokenizer, model = prepare_model_distillation(
            train_config, distil_config, fsdp_config, rank, kwargs)
    print(model)

    # Load Data
    if not train_config.distillation:
        steps_per_epoch, train_dataloader, eval_dataloader = get_dataloader(
            train_config, tokenizer, kwargs, rank)
    else:
        steps_per_epoch, train_dataloader, eval_dataloader = get_dataloader_distillation(
            train_config, student_tokenizer, teacher_tokenizer, kwargs, rank)

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=train_config.lr, epochs=train_config.num_epochs, steps_per_epoch=steps_per_epoch,
                                                    pct_start=train_config.pct_start, div_factor=train_config.div_factor, final_div_factor=train_config.final_div_factor)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        student_tokenizer if train_config.distillation else tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        generate_dataset_config(train_config, kwargs),
        steps_per_epoch,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank == 0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]


if __name__ == "__main__":
    fire.Fire(main)
