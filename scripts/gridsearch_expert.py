import gc
import argparse
import random

import bittensor
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

try:
    pile_dataset = load_dataset("the_pile", split="train")
    print("Successfully loaded the Pile dataset.")
except Exception as e:
    print(f"Error loading the Pile dataset: {e}")

alpha = 0.032

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--expert_name", type=str, required=True)

    return parser.parse_args()

def main(args):
    args = parse_args()
    run = wandb.init()

    with open(f"conf/{args.model_name}/architecture.yaml") as fh:
        makoto_config = yaml.load(fh, Loader=yaml.FullLoader)
    model_config = makoto_config["components"][args.expert_name]

    learning_rate = wandb.config.learning_rate
    num_batches = wandb.config.num_batches

    # Load the Pile dataset
    pile_dataset = load_dataset("the_pile", split="train")
    pile_dataset = pile_dataset.shuffle(seed=42)

    # Bittensor tokenizer
    bttokenizer = bittensor.tokenizer()

    model = AutoModelForCausalLM.from_pretrained(model_config["base_model"])
    base_tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    model = model.to("cuda")

    if wandb.config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif wandb.config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("optimizer not implemented")

    i = 0
    with tqdm(total=num_batches) as pbar:
        while i < num_batches:
            # Sample a batch from the Pile dataset
            batch = pile_dataset.select(range(i * wandb.config.batch_size, (i + 1) * wandb.config.batch_size))
            texts = [example['text'] for example in batch]
            # Use Bittensor tokenizer
            inputs = bttokenizer.batch_decode(texts)
            inputs = base_tokenizer(inputs, return_tensors="pt", padding=True)["input_ids"]

            inputs = inputs.to("cuda")
            model_inputs = dict()
            model_inputs["input_ids"] = inputs
            model_inputs["labels"] = inputs

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
    args = parse_args()
    wandb.agent(
        sweep_id=args.sweep_id,
        function=lambda: main(args),
        entity="skynetcc",
        project=f"{args.model_name}_experts",
    )