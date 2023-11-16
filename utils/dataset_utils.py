# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import importlib
from pathlib import Path

import torch

from data.concatenator import ConcatDataset

from utils.config_utils import generate_dataset_config, get_dataloader_kwargs

def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module


def get_custom_dataset(dataset_config, tokenizer, split: str):
    if not dataset_config.file:
        raise ValueError(f"Dataset not specified.")

    if dataset_config.file.endswith('.py'):
        module_path, func_name = Path(dataset_config.file), "get_custom_dataset"
    else:
        module_path, func_name = Path(dataset_config.file+"/load.py"), "get_custom_dataset"
        
    if not module_path:
        raise ValueError(f"Dataset not specified.")
    
    if not os.path.isfile(module_path):
        raise ValueError(f"The dataset folder doesn't contain a loading file (load.py).")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except AttributeError as e:
        print(f"It seems like the given method name ({func_name}) is not present in the load.py file ({module_path.as_posix()}).")
        raise e
    
def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )

    return get_custom_dataset(
        dataset_config,
        tokenizer,
        get_split(),
    )

def get_dataloader(train_config, tokenizer, kwargs, rank):
    dataset_config = generate_dataset_config(train_config, kwargs)

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(
            dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(
        train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(
                dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(
            train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    return train_dataloader, eval_dataloader

def get_dataloader_distillation(train_config, student_tokenizer, teacher_tokenizer, kwargs, rank):
    student_train_dataloader, student_eval_dataloader = get_dataloader(train_config, student_tokenizer, kwargs, rank)
    teacher_train_dataloader, teacher_eval_dataloader = get_dataloader(train_config, teacher_tokenizer, kwargs, rank)
    return len(student_train_dataloader), zip(student_train_dataloader, teacher_train_dataloader), zip(student_eval_dataloader, teacher_eval_dataloader)