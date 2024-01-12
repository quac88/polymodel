import json
from abc import ABC
from multiprocessing import shared_memory

import numpy as np
import requests
import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


def create_shared_block(data):

    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    # # Now create a NumPy array backed by shared memory
    np_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    np_array[:] = data[:]  # Copy the original data into shared memory
    return shm, np_array


def load_shared_mem(req):
    existing_shm = shared_memory.SharedMemory(name=req["shm_name"])
    np_array = np.ndarray(req["shape"], dtype=req["dtype"], buffer=existing_shm.buf)
    return existing_shm, np_array


class MakotoClient(GPT2LMHeadModel, ABC):
    def __init__(self):
        gpt = AutoModelForCausalLM.from_pretrained("gpt2")
        super().__init__(config=gpt.config)

    def forward(self, train=False, **kwargs):

        shm, _data = create_shared_block(kwargs["input_ids"].numpy())
        response = requests.post(
            "http://127.0.0.1:16868/forward",
            json={"shm_name": shm.name, "shape": _data.shape, "dtype": str(_data.dtype)},
        )
        shm.close()
        shm.unlink()

        response_dict = json.loads(response.content)
        response_shm, _data = load_shared_mem(response_dict)
        response_shm.unlink()
        logits = torch.tensor(_data)

        return CausalLMOutput(
            logits=logits,
        )