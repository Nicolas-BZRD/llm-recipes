import os
import yaml
import torch.distributed as dist

from pathlib import Path
from torch.distributed.fsdp import StateDictType
from models.checkpoint_handler import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint


def save_model(model, optimizer, step, train_config, distil_config, fsdp_config, rank):
    if train_config.enable_fsdp or distil_config.enable_fsdp:
        dist.barrier()
    path = fr"{train_config.output_dir}/{step+1}"
    try: os.mkdir(path)
    except: pass

    if train_config.use_peft:
        if rank == 0: print(f"We are about to save the PEFT modules")
        model.save_pretrained(path)
        if rank == 0: print(f"PEFT modules are saved in {path} directory")

    elif train_config.enable_fsdp:
        if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
            print("Saving the FSDP model checkpoints using FULL_STATE_DICT")
            save_model_checkpoint(model, optimizer, rank, path)

        elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
            if train_config.save_optimizer:
                print("Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                save_model_and_optimizer_sharded(model, rank, path, optim=optimizer)
            else:
                print("Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                save_model_and_optimizer_sharded(model, rank, path)

    else:
        if rank == 0:
            print(f"We are about to save the model")
            model.save_pretrained(path)
            print(f"Model are saved in {path} directory")

    if train_config.enable_fsdp or distil_config.enable_fsdp:
        dist.barrier()

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(
        train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(
        fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
        train_config.dist_checkpoint_root_folder
        + "/"
        + train_config.dist_checkpoint_folder
        + "-"
        + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir, 'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank == 0:
            print(f"training params are saved in {file_name}")