import argparse
import os
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from datasets import load_dataset
import wandb

alpha = 0.032

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--expert_name", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
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
    batch_size = model_config["batch_size"]

    # Load the RedPajama dataset
    red_pajama_dataset = load_dataset("togethercomputer/RedPajama-Data-1T")

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
            # Select random dataset and example
            chosen_split = random.choice(list(red_pajama_dataset.keys()))
            dataset_split = red_pajama_dataset[chosen_split]
            example = random.choice(dataset_split)
            text = example['text']

            # Process the data as needed for your model
            # Note: You might need to tokenize or preprocess 'text' before passing it to the model

            inputs = ...  # Add logic to preprocess 'text' for your model

            model_inputs = {
                "input_ids": inputs,
                "labels": inputs,
            }

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

if __name__ == "__main__":
    args = parse_args()
    main()
