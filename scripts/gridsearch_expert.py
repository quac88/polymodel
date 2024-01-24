import gc
import argparse
import random
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from datasets import load_dataset
import wandb

alpha = 0.032

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--expert_name", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()  # Moved the args parsing inside the main function
    run = wandb.init()

    with open(f"conf/{args.model_name}/architecture.yaml") as fh:
        makoto_config = yaml.load(fh, Loader=yaml.FullLoader)
    model_config = makoto_config["components"][args.expert_name]

    # Initialize the GPT-2 tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load the Hugging Face dataset
    red_pajama_dataset = load_dataset("togethercomputer/RedPajama-Data-1T", 'default')

    model = AutoModelForCausalLM.from_pretrained(model_config["base_model"])
    model = model.to("cuda")

    if wandb.config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    elif wandb.config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.learning_rate)
    else:
        raise ValueError("optimizer not implemented")

    num_batches = wandb.config.num_batches
    i = 0
    with tqdm(total=num_batches) as pbar:
        while i < num_batches:
            # Randomly choose a data split
            chosen_split = random.choice(list(red_pajama_dataset.keys()))
            dataset_split = red_pajama_dataset[chosen_split]

            # Randomly select an example from the dataset
            example = random.choice(dataset_split)
            text = example["text"]  # Assuming 'text' is the field containing the data

            # Tokenize the text using GPT-2 tokenizer
            inputs = gpt2_tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            inputs = inputs.to("cuda")
            model_inputs = {
                "input_ids": inputs["input_ids"],
                "labels": inputs["input_ids"]
            }

            model.train()
            out = model(**model_inputs)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss = out.loss.detach().item()
            if i > 0:
                loss_ema = (1 - alpha) * loss_ema + alpha * loss
            else:
                loss_ema = loss
            wandb.log({"loss": loss, "loss_ema": loss_ema})

            i += 1
            pbar.update(1)

            del inputs
            gc.collect()

    run.finish()

if __name__ == "__main__":
    main()  # Ensure main is called when script is executed
