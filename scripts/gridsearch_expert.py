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

    learning_rate = wandb.config.learning_rate
    num_batches = wandb.config.num_batches

    print( "Loading Datasets...")
    # Load datasets
    datasets = {}
    assigned_datasets = model_config["redpajama_subsets"]
    datasets = {}
    for subset in assigned_datasets:
        print(f"Loading subset: {subset}")
        datasets[subset] = load_dataset(
            "togethercomputer/RedPajama-Data-1T", subset, streaming=True, trust_remote_code=True
        )['train']

    # Inspect a few entries from each dataset
    for subset in assigned_datasets:
        print(f"Inspecting subset: {subset}")
        sample_stream = iter(datasets[subset])
        for _ in range(3):  # Inspect the first 3 entries
            try:
                sample_entry = next(sample_stream)
                print(f"Sample entry from {subset}: {sample_entry}")
            except StopIteration:
                print(f"No more data in subset: {subset}")
                break

    print("Datasets loaded.")

    # Load the model and set it to the GPU
    model = AutoModelForCausalLM.from_pretrained(model_config["base_model"]).to("cuda")
    # Initialize the GPT-2 tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Set the padding token to the eos token if it is None for GPT-2 tokenizer
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    # Load the base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])
    # Set the padding token to the eos token if it is None for base tokenizer
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    print("WandB configuration:", dict(wandb.config))

    # Initialize the optimizer
    if wandb.config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    elif wandb.config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.learning_rate)
    else:
        raise ValueError("optimizer not implemented")

    print("Starting training...")
    i = 0
    with tqdm(total=num_batches) as pbar:
        loss_ema = None
        while i < num_batches:
            chosen_dataset = random.choice(assigned_datasets)
            print(f"Selected dataset: {chosen_dataset}")
            dataset_stream = iter(datasets[chosen_dataset])
            try:
                print("Fetching data entry...")
                data_entry = next(dataset_stream)
                print(f"Data entry fetched: {data_entry}")  # Print a portion of the data entry for debugging
            except StopIteration:
                # Handle the end of the dataset stream
                print("End of dataset stream reached.")
                break

            # Assuming 'text' is a list of strings in each data_entry
            if 'text' not in data_entry:
                print("No text in data entry, skipping...")
                continue

            # Assuming 'text' is a list of strings in each data_entry
            texts = [data_entry['text']]
            print(f"Processing texts: {texts}")

            inputs = gpt2_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            # Prepare model inputs
            model_inputs = {
                "input_ids": inputs["input_ids"],
                "labels": inputs["input_ids"].clone()  # Assuming a language modeling task
            }

            print("Running model...")
            model.train()
            out = model(**model_inputs)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("Model run completed.")

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