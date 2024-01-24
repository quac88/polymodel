import argparse
import os
import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from makoto.server import MakotoResponse, MakotoServer
from makoto.utils import TokenizerWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_batches", type=int, default=15_000)
    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.00001)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists("models/"): os.makedirs("models")

    with open("conf/makoto_config.yaml", "r") as fh:
        config = yaml.safe_load(fh)
    assert os.getenv("MAKOTO_TRAIN") is not None, "MAKOTO_TRAIN not in environment."
    assert os.getenv("MAKOTO_TRAIN") in ["0", "1", "2"], "MAKOTO_TRAIN must be '0', '1' or '2'."
    model_save_path = os.path.join("models", config["model_name"])
    wandb.init(project="train-makoto-server", entity="skynetcc")

    try:
        model = torch.load(model_save_path)
    except FileNotFoundError:
        print(f"{config['model_name']} not found, creating it.")
        model = MakotoServer(config)

    print(model)
    print(f"Model has: {model.n_head_parameters:,} head parameters")
    print(f"Model has: {model.n_expert_parameters:,} expert parameters")
    print(f"Model has: {model.n_total_parameters:,} total parameters")

    # Load the RedPajama dataset
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T")
    # Make necessary adjustments if the format of RedPajama is different from the previous dataset
    tokenizer = TokenizerWrapper(args.sequence_length)
    dataset = dataset.map(tokenizer, batched=True, remove_columns=["text"])
    dataloader = DataLoader(dataset['train'], batch_size=args.batch_size)  # Assuming 'train' split is used

    for i, batch in tqdm(enumerate(dataloader), total=args.num_batches):
        if i == args.num_batches:
            break
        model_inputs = {
            "input_ids": batch["tokens"].to(model.device),
            "labels": batch["tokens"].to(model.device)
        }
        output: MakotoResponse = model(model_inputs, train=True)

        wandb_dict = {"loss/mixture": output.loss}
        for k in output.expert_losses.keys():
            wandb_dict[f"loss/{k}"] = output.expert_losses[k]
            wandb_dict[f"weight/{k}"] = output.expert_weights[k]
        wandb.log(wandb_dict)
        if i % 1000 == 0 and i != 0:
            model.save(model_save_path)
