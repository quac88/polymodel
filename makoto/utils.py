import bittensor
import torch
from torch.nn import CrossEntropyLoss

tokenizer = bittensor.tokenizer()


def causal_lm_loss(logits: torch.tensor, inputs: torch.tensor) -> torch.tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


class TokenizerWrapper(object):
    "cause its easier to do this than bother starmapping some thing to the dataloader class"

    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def _tokenize(self, example):
        text = example["text"]
        tokens = tokenizer(
            text,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]
        example["tokens"] = tokens
        return example

    def __call__(self, example):
        return self._tokenize(example)