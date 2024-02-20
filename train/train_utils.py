import os
import time
import torch
import wandb
import torch.distributed as dist

from tqdm import tqdm
from contextlib import nullcontext
from models.memory import MemoryTrace
from train.tools import clear_gpu_cache
from train.evaluations import evaluation
from train.save import save_train_params, save_model
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from models.distillation_model import DistillationLoss, preprocess_distillation_batch

def train(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, distil_config, dataset_config, teacher_train_dataloader=None, teacher_eval_dataloader=None, fsdp_config=None, local_rank=None, rank=None):
    # Weights & Biases tracking system initialization.
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    if rank == 0:
        wandb.init(
            project=f"llm_distillation_{dataset_config.file.split('/')[-1][:-3]}",
            name=f"{train_config.model_name.split('/')[-1]}-{model.teacher.name_or_path.split('/')[-1]}-d{distil_config.distil_factor}-t{distil_config.teacher_temperature}{distil_config.student_temperature}" if train_config.distillation else f"{train_config.model_name.split('/')[-1]}",
            config={
                "model_name": train_config.model_name.split('/')[-1],
                "dataset": dataset_config.file.split('/')[-1],
                "batch_size_training": train_config.batch_size_training,
                "val_batch_size": train_config.val_batch_size,
                "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
                "num_epochs": train_config.num_epochs,
                "lr": train_config.lr,
                "weight_decay": train_config.weight_decay,
                "pct_start": train_config.pct_start,
                "div_factor": train_config.div_factor,
                "final_div_factor": train_config.final_div_factor,
                "seed": train_config.seed,
                "use_fp16": train_config.use_fp16,
                "mixed_precision": train_config.mixed_precision,
                "peft_method": train_config.peft_method,
                "use_peft": train_config.use_peft,
                "freeze_layers": train_config.freeze_layers,
                "num_freeze_layers": train_config.num_freeze_layers,
                "quantization": train_config.quantization,
                "cross_entropy_factor": distil_config.cross_entropy_factor if train_config.distillation else -1,
                "distil_factor": distil_config.distil_factor if train_config.distillation else -1,
                "student_temperature": distil_config.student_temperature if train_config.distillation else -1,
                "teacher_temperature": distil_config.teacher_temperature if train_config.distillation else -1
            }
        )

    # Init distillation loss if distillation is enabled
    if train_config.distillation:
        distillation_loss = DistillationLoss(distillation_weight=distil_config.distil_factor, student_temperature=distil_config.student_temperature, teacher_temperature=distil_config.teacher_temperature, skip_student_eos=True, debug=True, debug_rank=0, tokenizer_student=model.student.name_or_path, tokenizer_teacher=model.teacher.name_or_path)

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
    val_ppl = []
    val_loss = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    steps_per_eval = len(eval_dataloader)
    steps_per_epoch = len(train_dataloader)
    best_val_loss = float("inf")
    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        total_length = steps_per_epoch//gradient_accumulation_steps
        model.student.train() if train_config.distillation else model.train()
        with MemoryTrace() as memtrace:
            total_loss = 0.0
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader if not train_config.distillation else zip(train_dataloader, teacher_train_dataloader)):
                if train_config.distillation: batch = preprocess_distillation_batch(batch)
                for key in batch.keys():
                    if train_config.enable_fsdp or distil_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to('cuda:0')

                with autocast():
                    if train_config.distillation:
                        student_output, teacher_output = model(**batch)
                        loss, cross_loss, dist_loss = distillation_loss(student_output, teacher_output, batch['student_labels'], batch['teacher_labels'], rank=rank)
                    else:
                        loss = model(**batch).loss

                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == steps_per_epoch - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        pbar.update()
                else:
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == steps_per_epoch - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update()

                if rank == 0:
                    if train_config.distillation:
                        wandb.log({
                            "train_loss": loss.detach().float(),
                            "cross_loss": cross_loss.detach().float(),
                            "distil_loss": dist_loss.detach().float(),
                            "teacher_loss": teacher_output.loss.detach().float(),
                            "lr": optimizer.param_groups[0]['lr']
                        })
                    else:
                        wandb.log({
                            "train_loss": loss.detach().float(),
                            "lr": optimizer.param_groups[0]['lr']
                        })

                lr_scheduler.step()
                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{steps_per_epoch} completed (loss: {loss.detach().float()})")

                if train_config.run_validation and ((step+1) % train_config.save_step == 0 or step+1 == steps_per_epoch):
                    if rank == 0: print("Running evaluation...")
                    model.eval()
                    eval_ppl, eval_epoch_loss, eval_cross_loss, eval_dist_loss = evaluation(
                        model, train_config, distil_config, 
                        eval_dataloader if not train_config.distillation else zip(eval_dataloader, teacher_eval_dataloader),
                        steps_per_eval, local_rank)
                    model.student.train() if train_config.distillation else model.train()
                    val_loss.append(eval_epoch_loss)
                    val_ppl.append(eval_ppl)
                    
                    if rank == 0:
                        print(f"Perplexity {eval_ppl}, loss {eval_epoch_loss}")
                        if train_config.distillation:
                            wandb.log({
                                "eval_ppl": eval_ppl,
                                "eval_epoch_loss": eval_epoch_loss,
                                "eval_cross_loss": eval_cross_loss,
                                "eval_dist_loss": eval_dist_loss
                            })
                        else:
                            wandb.log({
                                "eval_ppl": eval_ppl,
                                "eval_epoch_loss": eval_epoch_loss,
                            })

                    if eval_epoch_loss < best_val_loss or train_config.save_all:
                        if eval_epoch_loss < best_val_loss:
                            best_val_loss = eval_epoch_loss
                            if rank == 0:
                                print(f"best eval loss is {best_val_loss}")
                        if train_config.save_model:
                            checkpoint_start_time = time.perf_counter()
                            save_model(
                                model if not train_config.distillation else model.student, 
                                optimizer, ((steps_per_epoch*epoch)+step), train_config, distil_config, fsdp_config, rank
                            )
                            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                            checkpoint_times.append(checkpoint_end_time)
                    clear_gpu_cache(rank)
            pbar.close()
        
        if rank == 0: print(memtrace)
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)

        if torch.cuda.device_count() > 1 and train_config.enable_fsdp or distil_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / steps_per_epoch
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        if rank == 0:
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
        avg_eval_prep = sum(val_ppl)/len(val_ppl)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)

    return results