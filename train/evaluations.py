import os
import torch
import torch.distributed as dist

from tqdm import tqdm
from models.distillation_model import DistillationLoss, preprocess_distillation_batch

def evaluation(model, train_config, distil_config, eval_dataloader, steps_per_eval, local_rank):
    if train_config.enable_fsdp or distil_config.enable_fsdp: world_size = int(os.environ["WORLD_SIZE"])
    if train_config.distillation: distillation_loss = DistillationLoss(skip_student_eos=True)
    eval_loss = 0.0
    eval_cross_loss = 0.0
    eval_dist_loss = 0.0
    pbar = tqdm(colour="green", desc="Evaluating", total=steps_per_eval, dynamic_ncols=True)
    with torch.no_grad():
        for _, batch in enumerate(eval_dataloader):
            if train_config.distillation:
                batch = preprocess_distillation_batch(batch)
            for key in batch.keys():
                if train_config.enable_fsdp or distil_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')

            if train_config.distillation:
                outputs, teacher_output = model(**batch)
                loss, cross_loss, dist_loss = distillation_loss(outputs, teacher_output, batch['student_labels'], batch['teacher_labels'])
                eval_cross_loss += cross_loss.detach().float()
                eval_dist_loss += dist_loss.detach().float()
            else:
                outputs = model(**batch)
                loss = outputs.loss
            eval_loss += loss.detach().float()
            pbar.update()

    if torch.cuda.device_count() > 1 and train_config.enable_fsdp or distil_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        if train_config.distillation:
            dist.all_reduce(eval_cross_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(eval_dist_loss, op=dist.ReduceOp.SUM)

    eval_loss /= steps_per_eval
    eval_cross_loss /= steps_per_eval
    eval_dist_loss /= steps_per_eval
    if train_config.enable_fsdp or distil_config.enable_fsdp:
        eval_loss /= world_size
        eval_cross_loss /= world_size
        eval_dist_loss /= world_size
    eval_ppl = torch.exp(eval_cross_loss if train_config.distillation else eval_loss)
    
    return eval_ppl, eval_loss, eval_cross_loss, eval_dist_loss