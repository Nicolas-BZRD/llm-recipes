import os
import torch
import importlib

from pathlib import Path
from data.concatenator import ConcatDataset
from configs.configs_utils import get_dataloader_kwargs


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


def get_dataset(dataset_config, tokenizer, split: str) -> torch.utils.data.Dataset:
    if not dataset_config.file:
        raise ValueError(
            f"Dataset not specified. Please select a dataset path with the parameter '--dataset.file'.")

    if dataset_config.file.endswith('.py'):
        module_path, func_name = Path(dataset_config.file), "get_split"
    else:
        module_path, func_name = Path(
            dataset_config.file+"/load.py"), "get_split"

    if not os.path.isfile(module_path):
        raise ValueError(
            f"The load.py file in the dataset folder or the path to a python loading file doesn't exist. {module_path}")
    module = load_module_from_py_file(module_path.as_posix())

    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except:
        raise ValueError(
            f"It seems like the given method name ({func_name}) is not present in the load.py file ({module_path.as_posix()}).")


def get_dataloader(dataset_config, train_config, tokenizer, rank):
    dataset_train = get_dataset(
        dataset_config,
        tokenizer,
        split=dataset_config.train_split
    )
    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(
            dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(
        train_config, dataset_train, tokenizer, "train")
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        shuffle=False,
        **train_dl_kwargs,
    )
    if rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    if (train_config.run_validation):
        dataset_val = get_dataset(
            dataset_config,
            tokenizer,
            split=dataset_config.val_split,
        )

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
        if rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")
        return train_dataloader, eval_dataloader
    else:
        return train_dataloader, None


def get_distillation_dataloader(dataset_config, train_config, distil_config, student_tokenizer, teacher_tokenizer, rank):
    student_train_dataloader, student_eval_dataloader = get_dataloader(
        dataset_config, train_config, student_tokenizer, rank)
    
    dataset_config_teacher = dataset_config
    dataset_config_teacher.context, dataset_config_teacher.few_shot = distil_config.context, distil_config.few_shot
    teacher_train_dataloader, teacher_eval_dataloader = get_dataloader(
        dataset_config_teacher, train_config, teacher_tokenizer, rank)

    return student_train_dataloader, teacher_train_dataloader, student_eval_dataloader, teacher_eval_dataloader