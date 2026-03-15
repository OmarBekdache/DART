import math
import torch
import torch.nn as nn


class FedNova:
    """
    FedNova (Federated Normalized Averaging) — aggregation-only.

    Server update (conceptual):
      prev = θ^t
      Δ_k = (θ_k - prev)
      τ_k = (# local SGD steps client k took)
      w_k = n_k / sum_j n_j  (data weight)

      θ^{t+1} = prev + Σ_k w_k * (Δ_k / τ_k)

    We estimate:
      τ_k = ceil(n_k / batch_size_k) * local_epochs
      (or floor if drop_last=True)
    """

    def __init__(self, cfg, **_):
        self.cfg = cfg

    def build_client_loss(self, *, client_criterion, **_):
        # FedNova doesn't change the client objective.
        def loss_fn(outputs, targets, **__):
            return client_criterion(outputs, targets)
        return loss_fn

    def _get_batch_size_for_client(self, batch_size, client_idx: int) -> int:
        if isinstance(batch_size, int):
            return batch_size
        if isinstance(batch_size, (list, tuple)):
            return int(batch_size[client_idx])
        if isinstance(batch_size, dict):
            return int(batch_size[client_idx])
        raise TypeError("batch_size must be int, list/tuple, or dict keyed by client_idx")

    def aggregate(
        self,
        *,
        server_model,
        selected_clients_idx,
        per_client_data_samples,
        client_states,
        local_epochs: int,
        batch_size,
        drop_last: bool = False,
        **_,
    ):
        prev = server_model.state_dict()

        total = sum(per_client_data_samples[i] for i in selected_clients_idx)
        if total <= 0:
            raise ValueError("Total samples across selected clients must be > 0.")

        # Accumulate Σ w_k * (Δ_k / τ_k) in fp32
        delta_acc = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in prev.items()}

        with torch.no_grad():
            for i in selected_clients_idx:
                n_i = int(per_client_data_samples[i])
                if n_i <= 0:
                    continue

                bs_i = self._get_batch_size_for_client(batch_size, i)
                if bs_i <= 0:
                    raise ValueError(f"batch_size for client {i} must be > 0, got {bs_i}")

                # τ_i = (#steps per epoch) * local_epochs
                steps_per_epoch = (n_i // bs_i) if drop_last else math.ceil(n_i / bs_i)
                steps_per_epoch = max(1, int(steps_per_epoch))
                tau_i = steps_per_epoch * int(local_epochs)

                # FedAvg data weight
                w_i = n_i / float(total)

                csd = client_states[i]  # θ_i after local training

                for k in delta_acc:
                    # Only update floating-point tensors; keep buffers (e.g., BN counters) unchanged.
                    if not torch.is_floating_point(prev[k]):
                        continue

                    # Δ_i = θ_i - θ_prev
                    theta_prev = prev[k].detach().to(dtype=torch.float32, device=prev[k].device)
                    theta_i = csd[k].detach().to(dtype=torch.float32, device=prev[k].device)
                    delta = theta_i - theta_prev

                    # Accumulate w_i * (Δ_i / τ_i)
                    delta_acc[k].add_(delta, alpha=(w_i / float(tau_i)))

        # θ_new = θ_prev + delta_acc
        out = {}
        with torch.no_grad():
            for k, v_prev in prev.items():
                if torch.is_floating_point(v_prev):
                    updated = v_prev.detach().to(dtype=torch.float32, device=v_prev.device) + delta_acc[k].to(v_prev.device)
                    out[k] = updated.to(dtype=v_prev.dtype)
                else:
                    out[k] = v_prev  # keep non-float buffers as-is

        return out
