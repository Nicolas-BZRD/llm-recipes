import os
import time
import yaml
import wandb
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging


import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm


from model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from policies import fpSixteen, bfSixteen, get_llama_wrapper
from utils.memory_utils import MemoryTrace
from utils.distillation_model import distil_loss


def preprocess_distillation_batch(batch):
    batch_dict = {"student_" + key: value for key, value in batch[0].items()}
    batch_dict.update({"teacher_" + key: value for key,
                      value in batch[1].items()})
    return batch_dict

# Converting Bytes to Megabytes


def byte2mb(x):
    return int(x / 2**20)


def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, distil_config, dataset_config, steps_per_epoch, fsdp_config=None, local_rank=None, rank=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """

    # Weights & Biases tracking system initialization.
    if rank == 0:
        wandb.init(
            project=train_config.project_name,
            name=f"{train_config.model_name.split('/')[-1]}_{dataset_config.file.split('/')[-1]}_{int(time.time())}",
            config={
                "model_name": train_config.model_name.split('/')[-1],
                "dataset": dataset_config.file.split('/')[-1],
                "enable_fsdp": train_config.enable_fsdp,
                "low_cpu_fsdp": train_config.low_cpu_fsdp,
                "run_validation": train_config.run_validation,
                "batch_size_training": train_config.batch_size_training,
                "batching_strategy": train_config.batching_strategy,
                "context_length": train_config.context_length,
                "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
                "num_epochs": train_config.num_epochs,
                "num_workers_dataloader": train_config.num_workers_dataloader,
                "lr": train_config.lr,
                "weight_decay": train_config.weight_decay,
                "pct_start": train_config.pct_start,
                "div_factor": train_config.div_factor,
                "final_div_factor": train_config.final_div_factor,
                "seed": train_config.seed,
                "use_fp16": train_config.use_fp16,
                "mixed_precision": train_config.mixed_precision,
                "val_batch_size": train_config.val_batch_size,
                "peft_method": train_config.peft_method,
                "use_peft": train_config.use_peft,
                "output_dir": train_config.output_dir,
                "freeze_layers": train_config.freeze_layers,
                "num_freeze_layers": train_config.num_freeze_layers,
                "quantization": train_config.quantization,
                "one_gpu": train_config.one_gpu,
                "save_model": train_config.save_model,
                "dist_checkpoint_root_folder": train_config.dist_checkpoint_root_folder,
                "dist_checkpoint_folder": train_config.dist_checkpoint_folder,
                "save_optimizer": train_config.save_optimizer,
                "use_fast_kernels": train_config.use_fast_kernels,
            }
        )

    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp or distil_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = steps_per_epoch//gradient_accumulation_steps
            pbar = tqdm(
                colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader):
                if train_config.distillation:
                    batch = preprocess_distillation_batch(batch)
                for key in batch.keys():
                    if train_config.enable_fsdp or distil_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to('cuda:0')
                with autocast():
                    if train_config.distillation:
                        student_output, teacher_output = model(**batch)
                        loss = distil_loss(student_output, teacher_output, batch['student_labels'], batch['teacher_labels'])
                    else:
                        loss = model(**batch).loss
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == steps_per_epoch - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == steps_per_epoch - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update(1)

                if train_config.run_validation and ((step+1) % train_config.save_step == 0 or step+1 == steps_per_epoch):
                    eval_ppl, eval_epoch_loss = evaluation(
                        model, train_config, eval_dataloader, local_rank, tokenizer)
                    if rank == 0:
                        print(f" {eval_ppl} {eval_epoch_loss}")
                        wandb.log({
                            "eval_ppl": eval_ppl,
                            "eval_epoch_loss": eval_epoch_loss
                        })

                    checkpoint_start_time = time.perf_counter()
                    if train_config.save_model and eval_epoch_loss < best_val_loss:
                        if train_config.enable_fsdp:
                            dist.barrier()
                        if train_config.use_peft:
                            if train_config.enable_fsdp:
                                if rank == 0:
                                    print(
                                        f"we are about to save the PEFT modules")
                            else:
                                print(f"we are about to save the PEFT modules")
                            model.save_pretrained(train_config.output_dir)
                            if train_config.enable_fsdp:
                                if rank == 0:
                                    print(
                                        f"PEFT modules are saved in {train_config.output_dir} directory")
                            else:
                                print(
                                    f"PEFT modules are saved in {train_config.output_dir} directory")
                        elif fsdp_config:
                            if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                                save_model_checkpoint(
                                    model, optimizer, rank, train_config, epoch=epoch
                                )
                            elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                                print(
                                    " Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                print(
                                    "=====================================================")

                                save_model_and_optimizer_sharded(
                                    model, rank, train_config)
                                if train_config.save_optimizer:
                                    save_model_and_optimizer_sharded(
                                        model, rank, train_config, optim=optimizer)
                                    print(
                                        " Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                    print(
                                        "=====================================================")

                            if not train_config.use_peft and train_config.save_optimizer:
                                save_optimizer_checkpoint(
                                    model, optimizer, rank, train_config, epoch=epoch
                                )
                                print(
                                    " Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                print(
                                    "=====================================================")
                        else:
                            print(f"we are about to save the model")
                            model.save_pretrained(train_config.output_dir)
                            print(
                                f"Model are saved in {train_config.output_dir} directory")

                        if train_config.enable_fsdp:
                            dist.barrier()
                    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                    checkpoint_times.append(checkpoint_end_time)
                    if eval_epoch_loss < best_val_loss:
                        best_val_loss = eval_epoch_loss
                        if not train_config.enable_fsdp or (train_config.enable_fsdp and rank == 0):
                            print(f"best eval loss is {best_val_loss}")
                    val_loss.append(best_val_loss)
                    val_prep.append(eval_ppl)
                if rank == 0:
                    wandb.log({
                        "train_loss": loss.detach().float(),
                        "lr": optimizer.param_groups[0]['lr']
                    })
                lr_scheduler.step()
                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{steps_per_epoch} completed (loss: {loss.detach().float()})")
            pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / steps_per_epoch
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        if train_config.enable_fsdp:
            if rank == 0:
                print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                print(
                    f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                print(
                    f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                print(
                    f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(
                f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

        if not train_config.enable_fsdp or (train_config.enable_fsdp and rank == 0):
            print(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
            wandb.log({
                "train_perplexity": train_perplexity,
                "train_epoch_loss": train_epoch_loss,
                "train_epoch_time": epoch_end_time
            })

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(
        checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    # saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)

    return results


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            if train_config.distillation:
                batch = preprocess_distillation_batch(batch)
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                if train_config.distillation:
                    student_output, teacher_output = model(**batch)
                    loss = distil_loss(student_output, teacher_output, batch['student_labels'], batch['teacher_labels'])
                else:
                    outputs = model(**batch)
                    loss = outputs.loss
                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(
                    preds.detach().cpu().numpy(), skip_special_tokens=True)
            )

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    return eval_ppl, eval_epoch_loss


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


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        print(
            f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
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
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


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
