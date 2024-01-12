import gc
import argparse
import random

import bittensor
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from datasets import load_dataset

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--expert_name", type=str, required=True)
    return parser.parse_args()

# Main Function
def main(args):
    run = wandb.init()
    with open(f"conf/{args.model_name}/architecture.yaml") as fh:
        makoto_config = yaml.load(fh, Loader=yaml.SafeLoader)
    model_config = makoto_config["components"][args.expert_name]

    # wandb Configuration Checks
    learning_rate = wandb.config.learning_rate
    num_batches = wandb.config.num_batches
    batch_size = wandb.config.batch_size

    # Load and Shuffle Dataset
    try:
        pile_dataset = load_dataset("the_pile", split="train")
        pile_dataset = pile_dataset.shuffle(seed=42)
    except Exception as e:
        print(f"Error loading the Pile dataset: {e}")
        return

    # Model and Tokenizer Initialization
    model = AutoModelForCausalLM.from_pretrained(model_config["base_model"])
    base_tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Optimizer Setup
    if wandb.config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif wandb.config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("optimizer not implemented")

    # Training Loop
    i = 0
    loss_ema = None
    with tqdm(total=num_batches) as pbar:
        while i < num_batches:
            # Batch Selection
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > len(pile_dataset):
                break
            batch = pile_dataset.select(range(start_idx, end_idx))

            # Tokenization and Model Input Preparation
            texts = [example['text'] for example in batch]
            inputs = base_tokenizer(texts, return_tensors="pt", padding=True)["input_ids"]
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            model_inputs = {"input_ids": inputs, "labels": inputs}

            # Model Forward and Backward Passes
            model.train()
            out = model(**model_inputs)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Loss Tracking
            loss = out.loss.detach().item()
            loss_ema = (1 - alpha) * loss_ema + alpha * loss if loss_ema is not None else loss
            wandb.log({"loss": loss, "loss_ema": loss_ema})

            # Progress Update
            i += 1
            pbar.update(1)

            # Memory Management
            del inputs
            gc.collect()

    run.finish()

# Entry Point
if __name__ == "__main__":
    args = parse_args()
    wandb.agent(
        sweep_id=args.sweep_id,
        function=lambda: main(args),
        entity="skynetcc",
        project=f"{args.model_name}_experts",
    )
