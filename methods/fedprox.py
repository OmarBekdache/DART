import torch
import torch.nn as nn
from typing import Callable, Dict, Optional

class FedProx:
    def __init__(self, cfg, **_):
        self.cfg = cfg

    def build_client_loss(self, *, client_criterion, mu, server_model, **_):
        with torch.no_grad():
            global_params = {k: v.detach().clone() for k, v in server_model.state_dict().items()}

        def loss_fn(outputs, targets, *, model: Optional[nn.Module] = None, **_):
            m = model
            if m is None:
                raise ValueError(
                    "FedProx loss_fn needs a model to compute the proximal term. "
                    "Pass client_model to build_client_loss(...) or pass model=... to loss_fn."
                )

            base_loss = client_criterion(outputs, targets)

            # Proximal term: (mu/2) * sum ||w - w0||^2 over trainable params
            prox = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
            for name, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                # Align dtype/device; fall back to state_dict key if present
                if name in global_params:
                    g = global_params[name].to(p.device, dtype=p.dtype)
                else:
                    # In rare cases (e.g., different naming), try full state_dict key resolution
                    # or skip if not found.
                    continue
                prox = prox + (p - g).pow(2).sum()

            return base_loss + (mu * 0.5) * prox

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
