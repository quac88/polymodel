import argparse
import os

import bittensor
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import wandb

alpha = 0.032


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--expert_name", type=str, required=True)

    return parser.parse_args()


def main():
    if not os.path.exists("models"):
        os.makedirs("models")
    modelpath = os.path.join("models", args.model_name)
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    expert_path = os.path.join(modelpath, args.expert_name) + ".torch"

    wandb.init(entity="skynetcc", project=f"train_experts-{args.model_name}", name=args.expert_name)
    with open(f"conf/{args.model_name}/architecture.yaml") as fh:
        makoto_config = yaml.load(fh, Loader=yaml.FullLoader)
    model_config = makoto_config["components"][args.expert_name]

    learning_rate = model_config["learning_rate"]
    num_batches = model_config["num_batches"]

    dataset = bittensor.dataset(
        dataset_name=model_config["mountain_subsets"],
        block_size=256,
        batch_size=model_config["batch_size"],
    )

    model = AutoModelForCausalLM.from_pretrained(model_config["base_model"])
    model = model.to("cuda")

    if model_config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif model_config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("optimizer not implemented")

    i = 0
    with tqdm(total=num_batches) as pbar:
        while i < num_batches:
            inputs = next(dataset)
            inputs = inputs.to("cuda")
            model_inputs = dict()
            model_inputs["input_ids"] = inputs
            model_inputs["labels"] = inputs

            model.train()
            out = model(**model_inputs)
            out.loss.backward()
            try:
                optimizer.step()
            except Exception as e:
                print(e)
                print(f"In {args.expert_name}")
            optimizer.zero_grad()

            loss = out.loss.detach().item()
            if i > 0:
                loss_ema = (1 - alpha) * loss_ema + alpha * loss
            else:
                loss_ema = loss
            wandb.log({"loss": loss, "loss_ema": loss_ema})

            i += 1
            torch.cuda.empty_cache()
            pbar.update(1)
            if i % 1000 == 0:
                torch.save(model, expert_path)
    torch.save(model, expert_path)

    dataset.close()


if __name__ == "__main__":
    args = parse_args()
    main()