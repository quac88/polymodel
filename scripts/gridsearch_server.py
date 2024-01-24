import argparse
import os
import random
import yaml
from tqdm import tqdm
from transformers import enable_full_determinism
from datasets import load_dataset
import wandb
from makoto.server import MakotoResponse, MakotoServer

NUM_DATASETS_PER_EPOCH = (13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 6, 9, 13, 13, 13)
ema_alpha = 0.01

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    run = wandb.init()
    model_name = args.model_name

    # Load the RedPajama dataset
    red_pajama_dataset = load_dataset("togethercomputer/RedPajama-Data-1T")

    # Load and configure the model
    with open(f"conf/{model_name}/architecture.yaml", "r") as fh:
        config = yaml.safe_load(fh)
    assert os.getenv("MAKOTO_TRAIN") is not None, "MAKOTO_TRAIN not in environment."
    assert os.getenv("MAKOTO_TRAIN") in ["0", "1", "2"], "MAKOTO_TRAIN must be '0', '1' or '2'."

    learning_rate = wandb.config.learning_rate
    batches_per_dataset = wandb.config.batches_per_dataset
    batch_size = wandb.config.batch_size
    enc_dim = wandb.config.enc_dim
    use_transformer = wandb.config.use_transformer

    config["enc_dim"] = enc_dim
    enable_full_determinism(69)

    model = MakotoServer(config, learning_rate=learning_rate, use_transformer=use_transformer)
    print(f"Model has: {model.n_head_parameters:,} head parameters")
    print(f"Model has: {model.n_expert_parameters:,} expert parameters")
    print(f"Model has: {model.n_total_parameters:,} total parameters")

    ema_loss = 0
    for epoch in tqdm(range(len(NUM_DATASETS_PER_EPOCH)), desc="Epoch"):
        random.seed(69 + epoch)
        i = 0
        while i < batches_per_dataset:
            # Select random dataset and example
            chosen_split = random.choice(list(red_pajama_dataset.keys()))
            dataset_split = red_pajama_dataset[chosen_split]
            example = random.choice(dataset_split)
            text = example['text']

            # Prepare model inputs (assumes MakotoServer can handle raw text)
            model_inputs = {
                "input_ids": text,
                "labels": text,
            }
            output: MakotoResponse = model(model_inputs, train=True)

            ema_loss = ema_alpha * output.loss + (1 - ema_alpha) * ema_loss
            wandb_dict = {"loss/mixture": output.loss, "loss/mixture_ema": ema_loss}
            for k in output.expert_losses.keys():
                wandb_dict[f"loss/{k}"] = output.expert_losses[k]
                wandb_dict[f"weight/{k}"] = output.expert_weights[k]
            wandb.log(wandb_dict)

            i += 1

    run.finish()

if __name__ == "__main__":
    args = parse_args()
    sweep_id = args.sweep_id
    wandb.agent(
        sweep_id=sweep_id,
        function=main,
        entity="skynetcc",
        project=f"{args.model_name}_server",
    )
