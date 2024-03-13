# LLM-Recipes (Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs - Paper)

This project is a fork of the LLama recipes repository, which contains a collection of recipes and scripts for training and fine-tuning Large Language Models (LLMs). For comprehensive documentation and additional usage examples, please refer to the [LLama recipes repository](https://github.com/facebookresearch/llama-recipes).

This README file specifically focuses on distillation runs, which involve transferring knowledge from a teacher model to a student model using distillation techniques based on the paper "[Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs](https://arxiv.org/abs/2402.12030)".

HuggingFace implementation in progress...

## Run Distillation Process


For distillation, several parameters can be set:
- `--model_name`: The ID of the student model (HuggingFace repository ID).
- `--lr`: Learning rate for the training process.
- `--num_epochs`: Number of epochs for training.
- `--batch_size_training`: Batch size for training.
- `--val_batch_size`: Batch size for validation.
- `--dataset.file`: Path to the dataset file.
- `--output_dir`: Directory to save the output.

- `--distillation`: Activate distillation.
- `--distillation_config.model_name`: The ID of the teacher model (HuggingFace repository ID).
- `--distillation_config.enable_fsdp`: Enable Fully Sharded Data Parallelism (FSDP).
- `--distillation_config.pure_bf16`: Use pure BF16 precision.
- `--distillation_config.distil_factor`: Factor for distillation loss.
- `--save_step`: Interval for saving checkpoints during training.
- `--encoder_decoder`: Specify this parameter if the student model follows an encoder-decoder architecture.

## Example

Below is an example command for running the distillation process:

```bash
llm-recipes/finetuning.py --model_name EleutherAI/pythia-410m-deduped --dataset.file datasets/loader/squad.py --lr 1e-6 --num_epochs 5 --batch_size_training 4 --val_batch_size 4 --output_dir train/output/path --distillation_config.model_name meta-llama/Llama-2-7b-chat-hf --distillation --distillation_config.enable_fsdp --distillation_config.pure_bf16 --distillation_config.distil_factor 1.5 --save_step 100
```

In this example, the values for the parameters are replaced as follows:
- `model_name`: EleutherAI/pythia-410m-deduped
- `teacher_model_name`: meta-llama/Llama-2-7b-chat-hf
- `lr`: 1e-6
- `num_epochs`: 5
- `batch_size_training`: 4
- `val_batch_size`: 4
- `distil_factor`: 1.5
- `dataset`: llm-distillation/datasets/loader/squad.py

## Dataset File

In order to recalculate teacher logits correctly, the parameters used to generate them must be exactly the same as those used to create the dataset with the [LLM-Distillation library](https://github.com/Nicolas-BZRD/llm-distillation).

Example:
```python
import os
import sys
from datasets import load_from_disk

sys.path.append(f"{os.getenv('HOME')}/llm-distillation")
from prompt.prompt import create_chat_prompt
from prompt.prompt import create_prompt

def tokenize(item, tokenizer, encoder_decoder=False):
    is_chat = True if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower() else False
    task = "qa"

    if tokenizer.name_or_path == "meta-llama/Llama-2-7b-chat-hf":
        shot = 1
        title = False
    elif tokenizer.name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
        shot = 3
        title = item['title']
    elif tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
        shot = 4
        title = False

    if is_chat:
        prompt = create_chat_prompt(
            task, shot,
            title = title,
            context = item['context'],
            question = item['question'],
            sys_user = True if "mistralai/Mistral-7B-Instruct-v0.2" in tokenizer.name_or_path else False,
            chat_template = tokenizer.apply_chat_template
        )
    else:
        prompt = create_prompt(
            task, 0, 
            context = item['context'],
            question = item['question'],
        )

    context_tokens = tokenizer.encode(f"{tokenizer.bos_token} {prompt}", add_special_tokens=False)
    if not encoder_decoder:
        if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower():
            context_tokens = tokenizer.encode(f"{prompt}", add_special_tokens=False)
            if tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
                answer_tokens = tokenizer.encode(f" {item['answers_generated']}", add_special_tokens=False)
            else:
                answer_tokens = tokenizer.encode(f"{item['answers_generated']}", add_special_tokens=False)
        else:
            context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False)
            answer_tokens = tokenizer.encode(f" {item['answers_generated']}{tokenizer.eos_token}", add_special_tokens=False)

        prompt_tokens = context_tokens+answer_tokens
        labels_tokens = (len(context_tokens)*[-100,])+answer_tokens

        combined_tokens = {
            "input_ids": prompt_tokens,
            "labels": labels_tokens
        }
        return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))
    else:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")[0]
        labels = tokenizer.encode(item['answers_generated'], add_special_tokens=True, return_tensors="pt")[0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1]*len(input_ids)
        }

def get_split(dataset_config, tokenizer, split):
    dataset = load_from_disk(f"{os.getenv('HOME')}/llm-distillation/datasets/hf/{dataset_config.generated_by.split('/')[-1]}-squad")
    dataset = dataset[split]
    if dataset_config.training_size < 1: dataset = dataset.select(range(int(len(dataset)*dataset_config.training_size)))
    dataset = dataset.map(lambda item: tokenize(item, tokenizer, dataset_config.encoder_decoder), remove_columns=list(dataset.features))
    return dataset
```

## Citation

```
@misc{boizard2024crosstokenizer,
      title={Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs}, 
      author={Nicolas Boizard and Kevin El Haddad and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2402.12030},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
