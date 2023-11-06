# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import importlib
from functools import partial
from pathlib import Path

import torch

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
