import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from typing import Callable, Dict


class FedAugMix:
    def __init__(self, cfg, **_):
        self.cfg = cfg
    
    def build_client_loss(self, *, client_criterion, **_):
        if self.cfg.no_jsd:
            def loss_fn(outputs, targets, **_):
                return client_criterion(outputs, targets)
        else:
            def loss_fn(outputs, targets, *, inputs, **_):
                outputs_clean, outputs_aug1, outputs_aug2 = torch.split(outputs, inputs[0].size(0))
                loss = client_criterion(outputs_clean, targets)
                p_clean, p_aug1, p_aug2 = F.softmax(outputs_clean, dim=1), F.softmax(outputs_aug1, dim=1), F.softmax(outputs_aug2, dim=1)
                p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
                return loss
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