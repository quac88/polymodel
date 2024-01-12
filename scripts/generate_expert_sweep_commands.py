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
    with open(os.path.join("conf", model_name, "expert_sweep.yaml"), "r") as fh:
        sweep_config = yaml.safe_load(fh)

    return model_arch, sweep_config


def main():
    args = parse_args()

    model_arch, sweep_config = load_configs(args.model_name)

    final_command = ""
    for i, component in enumerate(model_arch["components"].keys()):
        sweep_config["name"] = component
        sweep_id = wandb.sweep(
            sweep=sweep_config, project=f"{args.model_name}_experts", entity="skynetcc"
        )

        pm2_command = (
            f"CUDA_VISIBLE_DEVICES={i} pm2 start scripts/gridsearch_expert.py"
            f" --name {component} --no-autorestart --"
            f" --sweep_id {sweep_id}"
            f" --model_name {args.model_name} --expert_name {component}"
        )
        final_command += pm2_command + " && "
    print(final_command)


if __name__ == "__main__":
    main()