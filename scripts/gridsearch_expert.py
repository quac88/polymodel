import gc
import argparse
import random
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GPT2Tokenizer, AutoTokenizer
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
    run = wandb.init(project="ChocolateCake", entity="skynetcc")

    # Load configuration from the YAML file
    with open(f"/root/mac/polymodel/conf/{args.model_name}/architecture.yaml") as fh:
        makoto_config = yaml.load(fh, Loader=yaml.FullLoader)
    # Select the component configuration
    model_config = makoto_config["components"][args.expert_name]

    # Update the wandb configuration with the values from the YAML file
    wandb.config.update({
        "learning_rate": model_config["learning_rate"],
        "num_batches": model_config["num_batches"],
        "optimizer": model_config["optimizer"],
        "batch_size": model_config["batch_size"],
    })
    learning_rate = model_config["learning_rate"]
    num_batches = model_config["num_batches"]

    # assign the datasets
    assigned_datasets = model_config["redpajama_subsets"]
    # Load the Hugging Face dataset
    red_pajama_dataset = load_dataset("togethercomputer/RedPajama-Data-1T", 'default', streaming=True)
    # Initialize the GPT-2 tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_config["base_model"])
    # Load the tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])
    # Set the padding token to the eos token if it is None
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    # Set the model to the GPU
    model = model.to("cuda")

    print("WandB configuration:", dict(wandb.config))

    # Initialize the optimizer
    if wandb.config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    elif wandb.config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.learning_rate)
    else:
        raise ValueError("optimizer not implemented")

    i = 0
    with tqdm(total=num_batches) as pbar:
        while i < num_batches:
            chosen_dataset = random.choice(assigned_datasets)
            inputs = next(red_pajama_dataset[chosen_dataset])
            texts = gpt2_tokenizer.batch_decode(inputs)
            inputs = gpt2_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
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
        function=main,
        entity="skynetcc",
        project=f"{args.model_name}",
    )
