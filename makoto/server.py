import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, NamedTuple, Sequence

import numpy as np
import torch
from bittensor import tokenizer as bttokenizer
from torch import Tensor as T
from torch import nn

from makoto.embeddings import ExpertEmbeddings, SentenceEmbedder
from makoto.utils import causal_lm_loss


from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        assert num_layers % 2 == 0, "Must specify an even number of layers for transformer"
        num_repeat_attention = num_layers // 2

        self.config = GPTNeoConfig(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            attention_types=[[["global", "local"], num_repeat_attention]],
            max_position_embeddings=2048,
            resid_dropout=0,
            embed_dropout=0,
            attention_dropout=0,
        )

        self.blocks = nn.ModuleList(
            GPTNeoBlock(config=self.config, layer_id=i) for i in range(self.config.num_layers)
        )

    def forward(self, inputs: torch.Tensor):

        hidden_state = inputs
        for block in self.blocks:
            hidden_state = block(hidden_state)[0]
        return hidden_state


class MakotoResponse(NamedTuple):
    loss: float
    logits: torch.Tensor
    hidden_states: torch.Tensor
    expert_losses: Dict[str, float]
    expert_weights: Dict[str, float]


class MakotoServer(nn.Module):
    def __init__(self, cfg: dict, learning_rate: float = 1e-5,
                 use_transformer: bool = False):
        super().__init__()

        self.name = cfg["model_name"]
        self.enc_dim = cfg["enc_dim"]
        self.device = f"cuda:{cfg['gpu']}"
        self.tokenizer = bttokenizer()

        # Constituent models & their projection layers.
        self.models = nn.ModuleDict()
        self.encoders_1 = nn.ModuleDict()
        for name, component in cfg["components"].items():
            device = f"cuda:{component['gpu']}"
            modelpath = os.path.join("models", self.name, name + ".torch")
            print("loading", modelpath)
            self.models[name] = torch.load(modelpath, map_location=device)
            if component["precision"] == "half":
                self.models[name].half()

            in_dim = self.models[name].transformer.ln_f.normalized_shape[0]
            self.encoders_1[name] = nn.Linear(in_dim, self.enc_dim)
        self.model_names = sorted(list(self.models.keys()))

        # Post-gating projection to logits.
        self.norm1 = nn.LayerNorm(self.enc_dim)
        self.act = torch.nn.ReLU()

        if use_transformer is True:
            self.encoder_2 = TransformerLayer(hidden_size=self.enc_dim)
        else:
            self.encoder_2 = nn.Linear(self.enc_dim, self.enc_dim)

        self.norm2 = nn.LayerNorm(self.enc_dim)
        self.decoder = nn.Linear(self.enc_dim, 50258, bias=False)  # lol hardcoding, fuck it

        # Routing tools.
        self.sequence_embedder = SentenceEmbedder()
        self.expert_embeddings = ExpertEmbeddings(
            model_names=self.model_names,
            embedding_dim=self.sequence_embedder.embedding_dimension,
        )

        self._go_to_gpu()
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _go_to_gpu(self):
        self.encoders_1.to(self.device)
        self.act.to(self.device)
        self.encoder_2.to(self.device)
        self.decoder.to(self.device)
        self.sequence_embedder.to(self.device)
        self.expert_embeddings.to(self.device)
        self.norm1.to(self.device)
        self.norm2.to(self.device)

    @property
    def n_head_parameters(self):
        return sum(
            [
                np.prod(p[1].size())
                for p in self.named_parameters()
                if not any(model_name in p[0] for model_name in self.model_names)
            ]
        )

    @property
    def n_expert_parameters(self):
        return sum(
            np.prod(p[1].size())
            for p in self.named_parameters()
            if any(model_name in p[0] for model_name in self.model_names)
        )

    @property
    def n_total_parameters(self):
        return sum(np.prod(p.size()) for p in self.parameters())

    def _get_expert_sequence_similarities(
        self, models: Sequence[str], sequences: Sequence[str]
    ) -> T:

        model_embeddings = self.expert_embeddings(models)
        sequence_embeddings = self.sequence_embedder(sequences)

        similarity_matrix = torch.mm(sequence_embeddings, model_embeddings.T)

        return nn.Softmax(dim=1)(similarity_matrix)

    def _local_forward(self, model_name: str, inputs, train=False):

        torch_model = self.models[model_name]

        if os.getenv("MAKOTO_TRAIN") in ("0", "1") or train is False:
            with torch.no_grad():
                outs = torch_model(output_hidden_states=True, **inputs)
        elif os.getenv("MAKOTO_TRAIN") in ("2") and train is True:
            outs = torch_model(output_hidden_states=True, **inputs)

        return outs

    def _clone_input_tensors_to_model_devices(self, data_dict):

        model_inputs_dict = dict()
        for model_name in self.model_names:
            model = self.models[model_name]
            model_inputs_dict[model_name] = dict()
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    model_inputs_dict[model_name][k] = v.to(model.device)
                else:
                    model_inputs_dict[model_name][k] = v

        return model_inputs_dict

    def forward(self, data, train=False) -> MakotoResponse:

        sequences = [self.tokenizer.decode(item) for item in data["input_ids"]]
        model_inputs = self._clone_input_tensors_to_model_devices(data)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                model_name: executor.submit(
                    self._local_forward, model_name, model_inputs[model_name], train
                )
                for model_name in self.model_names
            }

            outputs = {k: v.result() for k, v in futures.items()}

        logit_preds = []
        projected_states = []
        context_similarities = self._get_expert_sequence_similarities(self.model_names, sequences)

        mean_weights = context_similarities.mean(axis=0)
        expert_losses = dict()
        expert_weights = dict()
        for idx, model_name in enumerate(self.model_names):
            # idk why this is 0, but: https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L757
            # It returns a 13 length tuple for a sequence length of 4 idk wtf this is.
            hidden = outputs[model_name].hidden_states[-1].to(self.device).to(torch.float32)
            logits = outputs[model_name].logits.to(self.device)
            logit_preds.append(logits)

            weight = mean_weights[idx].item()
            expert_losses[model_name] = outputs[model_name].loss.item()
            expert_weights[model_name] = weight
            projected_states.append(self.encoders_1[model_name](hidden))

        # TODO: Rethink gating layer architecture cause we can just stack the projected tensors.
        projected_states = torch.stack(projected_states)
        # projected_states = torch.cat(projected_states, dim=-1)
        # projected_states = self.act(projected_states)

        context_similarities = context_similarities.unsqueeze(1)

        projected_states = projected_states.transpose(0, 1)
        batch_size, experts, seq_length, h_dim = projected_states.shape
        projected_states = projected_states.view(batch_size, experts, -1)
        projected_states = torch.bmm(context_similarities, projected_states).squeeze()
        projected_states = projected_states.reshape(batch_size, seq_length, h_dim)

        projected_states = self.norm1(projected_states)
        projected_states = self.act(projected_states)

        projected_states = self.encoder_2(projected_states)
        projected_states = self.norm2(projected_states)
        projected_states = self.act(projected_states)

        logits = self.decoder(projected_states)
        loss = causal_lm_loss(logits, data["labels"].to(self.device))

        if os.getenv("MAKOTO_TRAIN") in ("1", "2") and train is True:
            self.train()
            if os.getenv("MAKOTO_TRAIN") == "1":
                for name, model in self.models.items():
                    model.eval()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return MakotoResponse(
            loss=loss.detach(),
            logits=logits.detach(),
            hidden_states=projected_states.detach(),
            expert_weights=expert_weights,
            expert_losses=expert_losses,
        )

    def save(self, save_path: str):
        torch.save(self, save_path)