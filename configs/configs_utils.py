import torch.distributed as dist

from dataclasses import asdict
from transformers import default_data_collator
from torch.utils.data import DistributedSampler
from transformers.data import DataCollatorForSeq2Seq
from configs import lora_config, llama_adapter_config, prefix_config
from peft import LoraConfig, AdaptionPromptConfig, PrefixTuningConfig
from data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler

def update_config(config, isSubmodule=False, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, isSubmodule, **kwargs)
    else:
        for k, v in kwargs.items():
            if "." in k:
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
            elif not isSubmodule and hasattr(config, k):
                setattr(config, k, v)

def generate_peft_config(train_config, kwargs):
    configs = (lora_config, llama_adapter_config, prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple(c.__name__.rstrip("_config") for c in configs)

    assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}"

    config = configs[names.index(train_config.peft_method)]()

    update_config(config, **kwargs)
    params = asdict(config)
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)
    return peft_config

def get_dataloader_kwargs(train_config, dataset, tokenizer, mode, distil_config=None):
    fsdp = train_config.enable_fsdp or distil_config.enable_fsdp if distil_config else train_config.enable_fsdp
    kwargs = {}
    batch_size = train_config.batch_size_training if mode == "train" else train_config.val_batch_size
    if train_config.batching_strategy == "padding":
        if fsdp:
            kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                dataset,
                batch_size=batch_size,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode == "train",
                seed=train_config.seed
            )
        else:
            kwargs["batch_sampler"] = LengthBasedBatchSampler(
                dataset, batch_size, drop_last=True, shuffle=mode == "train", seed=train_config.seed)
        kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
    elif train_config.batching_strategy == "packing":
        if fsdp:
            kwargs["sampler"] = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode == "train",
                seed=train_config.seed
            )
        kwargs["batch_size"] = batch_size
        kwargs["drop_last"] = True
        kwargs["collate_fn"] = default_data_collator
    else:
        raise ValueError(
            f"Unknown batching strategy: {train_config.batching_strategy}")
    return kwargs