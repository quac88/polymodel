import argparse
import os

import yaml

import wandb


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)

    return parser.parse_args()


def load_configs(model_name: str):
    with open(os.path.join("conf", model_name, "architecture.yaml"), "r") as fh:
        model_arch = yaml.safe_load(fh)
    with open(os.path.join("conf", model_name, "server_sweep.yaml"), "r") as fh:
        sweep_config = yaml.safe_load(fh)

    return model_arch, sweep_config


def main():
    args = parse_args()

    model_arch, sweep_config = load_configs(args.model_name)

    sweep_id = wandb.sweep(
        sweep=sweep_config, project=f"{args.model_name}_server", entity="skynetcc"
    )
    python_cmd = (
        f"python -m scripts.gridsearch_server "
        f"--sweep_id {sweep_id} --model_name {args.model_name}"
    )
    with open("server_cmd.sh", "w") as fh:
        fh.write(python_cmd)

    print(f"Python command is: ")
    print(python_cmd)
    print()

    print("WANDB_DISABLE_SERVICE=True MAKOTO_TRAIN=1 CUDA_VISIBLE_DEVICES=")
    print(f"pm2 start server_cmd.sh --name ")


if __name__ == "__main__":
    main()