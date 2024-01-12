import argparse
import os
import random

import bittensor
import yaml
from tqdm import tqdm
from transformers import enable_full_determinism

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
    run = wandb.init()
    model_name = args.model_name

    # Create/load the mixture model.
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
    for epoch, num_datasets in tqdm(enumerate(NUM_DATASETS_PER_EPOCH), desc="Epoch"):
        random.seed(69 + epoch)
        i = 0
        selected_datasets = random.sample(bittensor.__datasets__, k=num_datasets)
        dataset = bittensor.dataset(
            block_size=264,
            batch_size=batch_size,
            dataset_name=selected_datasets,
            save_dataset=True,
        )
        pbar = tqdm(total=batches_per_dataset, desc=",".join(selected_datasets))
        while i < batches_per_dataset:
            inputs = next(dataset)
            model_inputs = {
                "input_ids": inputs.to(model.device),
                "labels": inputs.to(model.device),
            }
            output: MakotoResponse = model(model_inputs, train=True)

            ema_loss = ema_alpha * output.loss + (1 - ema_alpha) * ema_loss
            wandb_dict = {"loss/mixture": output.loss, "loss/mixture_ema": ema_loss}
            for k in output.expert_losses.keys():
                wandb_dict[f"loss/{k}"] = output.expert_losses[k]
                wandb_dict[f"weight/{k}"] = output.expert_weights[k]
            wandb.log(wandb_dict)

            pbar.update(1)
            i += 1
        dataset.close()

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