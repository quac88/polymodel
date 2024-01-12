import io
import os
import pickle
from copy import deepcopy
from multiprocessing import shared_memory
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from flask import Flask, request, send_file

import wandb
from makoto.server import MakotoResponse, MakotoServer

app = Flask("Makoto Server Backend")


def load_shared_mem(req):
    existing_shm = shared_memory.SharedMemory(name=req["shm_name"])
    np_array = np.ndarray(req["shape"], dtype=req["dtype"], buffer=existing_shm.buf)
    return existing_shm, np_array


def create_shared_block(data):

    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    # # Now create a NumPy array backed by shared memory
    np_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    np_array[:] = data[:]  # Copy the original data into shared memory
    return shm, np_array


def compress_logits(logits: torch.Tensor, k=4096) -> torch.Tensor:
    topk = torch.topk(logits, k=k, dim=-1, largest=True)
    compressed = torch.zeros_like(logits)
    compressed = compressed.scatter(dim=2, index=topk.indices, src=topk.values)
    return compressed.to("cpu").to_sparse()


def prepare_response(old_response: MakotoResponse):

    return SimpleNamespace(
        loss=old_response.loss.to("cpu"),
        logits=compress_logits(old_response.logits),
        hidden_states=old_response.hidden_states.to("cpu"),
        expert_losses=old_response.expert_losses,
        expert_weights=old_response.expert_weights,
    )


@app.route("/forward", methods=["POST"])
def forward():

    req = request.json
    global model

    _shm, inputs = load_shared_mem(req)

    data = {"input_ids": torch.tensor(inputs), "labels": torch.tensor(inputs)}
    if "output_hidden_states" in data.keys():
        del data["output_hidden_states"]
    if "labels" not in data.keys():
        data["labels"] = data["input_ids"]
    output = model(data, train=True)

    outstr1 = ""
    outstr2 = ""
    loss_dict = deepcopy(output.expert_losses)
    for k, v in sorted(output.expert_losses.items()):
        outstr1 += f" | {k:<13} |"
        outstr2 += f" | {v:<13.4f} |"

    k = "Mixture"
    v = output.loss.item()
    loss_dict["Mixture"] = v
    outstr1 += f" | {k:<13} |"
    outstr2 += f" | {v:<13.4f} |"
    print(outstr1)
    print(outstr2)
    wandb_dict = {f"loss/{k}": v for k, v in loss_dict.items()}
    wandb.log(wandb_dict)

    shm, _array = create_shared_block(output.logits.cpu().numpy())
    response = {"shm_name": shm.name, "shape": _array.shape, "dtype": str(_array.dtype)}

    return response


@app.route("/train_step", methods=["POST"])
def train_step():

    data = pickle.loads(request.data)
    global model
    logits = model(data, train=True)

    serialized = pickle.dumps(logits)
    stream = io.BytesIO(serialized)

    return send_file(stream, mimetype="application/zip")


@app.route("/save", methods=["POST"])
def save():
    return "not saved"


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    with open("conf/ChocolateCake/architecture.yaml", "r") as fh:
        config = yaml.safe_load(fh)
    assert os.getenv("MAKOTO_TRAIN") is not None, "MAKOTO_TRAIN not in environment."
    assert os.getenv("MAKOTO_TRAIN") in ["0", "1", "2"], "MAKOTO_TRAIN must be '0', '1' or '2'."

    global model
    try:
        model = torch.load("models/ChocolateCake/makoto_whole-meadow-10.torch")
    except FileNotFoundError:
        print(f"{config['model_name']} not found, creating it.")
        model = MakotoServer(config)

    assert model.name == config["model_name"], "Model name does not match config."

    print(model)
    print(f"Model has: {model.n_head_parameters:,} head parameters")
    print(f"Model has: {model.n_expert_parameters:,} expert parameters")
    print(f"Model has: {model.n_total_parameters:,} total parameters")

    wandb.init(project="live-server-makoto", entity="skynetcc")

    app.run(port=16868, threaded=True)