import torch
import torch.nn as nn

from config import Config
from typing import Callable, Dict

import torch

class FedAVG:
    def __init__(self, cfg, **_):
        self.cfg = cfg
    
    def build_client_loss(self, *, client_criterion, **_):
        def loss_fn(outputs, targets, **_):
            return client_criterion(outputs, targets)
        return loss_fn

    def aggregate(self, *, server_model, selected_clients_idx, per_client_data_samples, client_states, **_):
        total = sum(per_client_data_samples[i] for i in selected_clients_idx)
        out = {k: torch.zeros_like(v, dtype=torch.float) for k, v in server_model.state_dict().items()}

        with torch.no_grad():
            for i in selected_clients_idx:
                w = per_client_data_samples[i] / total
                csd = client_states[i]
                for k in out:
                    out[k].add_(csd[k], alpha=w)
        return out