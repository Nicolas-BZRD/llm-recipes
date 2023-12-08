import os
import sys
import yaml
import fire

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from models.checkpoint_handler import load_sharded_model_single_gpu

def load_model_from_config(config_path):
    model_config = AutoConfig.from_pretrained(config_path) 
    model = AutoModelForCausalLM.from_config(config=model_config)
    return model

def main(
    fsdp_checkpoint_path="", # Path to FSDP Sharded model checkpoints
    consolidated_model_path="", # Path to save the HF converted model checkpoints
    HF_model_path_or_name="" # Path/ name of the HF model that include config.json and tokenizer_config.json (e.g. meta-llama/Llama-2-7b-chat-hf)
    ):

    try:
        file_name = 'train_params.yaml'
        train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
        with open(train_params_path, 'r') as file:
            data = yaml.safe_load(file)

            HF_model_path_or_name = data.get('model_name')

            print(f"Model name: {HF_model_path_or_name}")
    except FileNotFoundError:
        if not HF_model_path_or_name:
            print(f"The file {train_params_path} does not exist.")
            HF_model_path_or_name = input("Please enter the model name: ")
            print(f"Model name: {HF_model_path_or_name}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
    model_def = load_model_from_config(HF_model_path_or_name)
    print("model is loaded from config")
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    print("model is loaded from FSDP checkpoints")
    tokenizer = AutoTokenizer.from_pretrained(HF_model_path_or_name)
    tokenizer.save_pretrained(consolidated_model_path)
    model.save_pretrained(consolidated_model_path)
    print(f"HuggingFace model checkpoints has been saved in {consolidated_model_path}")
if __name__ == "__main__":
    fire.Fire(main)
