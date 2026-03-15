import torch
import torch.nn as nn
from typing import Optional


class FedDyn:
    """
    Implements FedDyn (Acar et al., ICLR 2021) in the same "plugin" style as your FedProx class.

    Key state:
      - self.h: server-side vector (same structure as state_dict) updated every round
      - self.client_grads[i]: per-client stored gradient estimate ∇L_k(θ_k) (same structure)

    Local objective (for client k at round t), matching Algorithm 1 / Eq. (1):
        min_θ  L_k(θ) - <g_k^{t-1}, θ> + (alpha/2) ||θ - θ_server^{t-1}||^2
    with gradient-estimate update:
        g_k^t = g_k^{t-1} - alpha * (θ_k^t - θ_server^{t-1})

    Server updates:
        h^t = h^{t-1} - alpha*(1/m) * sum_{k in P_t} (θ_k^t - θ_server^{t-1})
        θ_server^t = avg_{k in P_t}(θ_k^t) - (1/alpha) * h^t
    """

    def __init__(self, cfg, **_):
        self.cfg = cfg
        self.h = None  # server-side vector, initialized lazily to zeros_like(server_model.state_dict())
        self.client_grads = {}  # client_idx -> dict(param_name -> tensor)

    def _zeros_like_state(self, state_dict, *, device=None, dtype=torch.float):
        out = {}
        for k, v in state_dict.items():
            t = v.detach()
            if device is not None:
                t = t.to(device)
            out[k] = torch.zeros_like(t, dtype=dtype)
        return out

    def build_client_loss(
        self,
        *,
        client_criterion,
        mu,  # we'll treat mu as "alpha" (FedDyn paper notation)
        server_model,
        client_idx=None,
        **_,
    ):
        if client_idx is None:
            raise ValueError("FedDyn.build_client_loss(...) requires client_idx=...")

        alpha = float(mu)

        # Snapshot server params θ^{t-1}
        with torch.no_grad():
            server_params = {k: v.detach().clone() for k, v in server_model.state_dict().items()}

        # Ensure per-client gradient estimate exists (g_k^{t-1})
        if client_idx not in self.client_grads:
            self.client_grads[client_idx] = self._zeros_like_state(server_params, dtype=torch.float)

        g_prev = self.client_grads[client_idx]  # dict

        def loss_fn(outputs, targets, *, model: Optional[nn.Module] = None, **__):
            m = model
            if m is None:
                raise ValueError(
                    "FedDyn loss_fn needs a model to compute regularization terms. "
                    "Pass model=... to loss_fn."
                )

            base_loss = client_criterion(outputs, targets)

            # FedDyn terms:
            #   - <g_prev, θ>  (linear term, note the minus in objective)
            #   + (alpha/2) ||θ - θ_server||^2
            lin = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
            quad = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)

            for name, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in server_params:
                    continue

                sp = server_params[name].to(p.device, dtype=p.dtype)
                gp = g_prev[name].to(p.device, dtype=p.dtype)

                # linear: - <g_prev, θ>
                lin = lin - (gp * p).sum()
                # quadratic: (alpha/2) ||θ - θ_server||^2
                quad = quad + (p - sp).pow(2).sum()

            return base_loss + lin + (alpha * 0.5) * quad

        return loss_fn

    def aggregate(
        self,
        *,
        server_model,
        selected_clients_idx,
        per_client_data_samples,
        client_states,
        mu,
        **_,
    ):
        
        alpha = float(mu)
        if alpha <= 0:
            raise ValueError("FedDyn.aggregate needs positive mu (used as alpha).")

        m = getattr(self.cfg, "num_clients", None)
        if m is None:
            m = len(client_states)

        prev = server_model.state_dict()
        if self.h is None:
            self.h = self._zeros_like_state(prev, dtype=torch.float)

        # ---- 1) Compute weighted average of participating client models (FedAvg-style) ----
        total = sum(per_client_data_samples[i] for i in selected_clients_idx)
        avg = {k: torch.zeros_like(v, dtype=torch.float) for k, v in prev.items()}

        with torch.no_grad():
            for i in selected_clients_idx:
                w = per_client_data_samples[i] / total
                csd = client_states[i]
                for k in avg:
                    avg[k].add_(csd[k], alpha=w)

        # ---- 2) Update per-client gradient estimates g_k (Algorithm 1 line 219-232) ----
        # g_k^t = g_k^{t-1} - alpha*(θ_k^t - θ^{t-1})
        #alpha = float(getattr(self.cfg, "mu", None) or getattr(self.cfg, "alpha", None) or 0.0)
        # If your runner passes mu via cfg, this picks it up. If not, set cfg.alpha or cfg.mu.
        # (We still accept mu in build_client_loss, but aggregate also needs alpha.)

        if alpha <= 0:
            raise ValueError(
                "FedDyn.aggregate needs a positive alpha. "
                "Set cfg.alpha or cfg.mu (and pass the same value as mu to build_client_loss)."
            )

        with torch.no_grad():
            for i in selected_clients_idx:
                if i not in self.client_grads:
                    self.client_grads[i] = self._zeros_like_state(prev, dtype=torch.float)
                gk = self.client_grads[i]
                csd = client_states[i]
                for k in prev:
                    delta = (csd[k].detach().float() - prev[k].detach().float())
                    gk[k].add_(delta, alpha=-alpha)  # g -= alpha * delta

        # ---- 3) Update server h (Algorithm 1 line 252-263) ----
        # h^t = h^{t-1} - alpha*(1/m) * sum_{k in P_t}(θ_k^t - θ^{t-1})
        with torch.no_grad():
            for k in self.h:
                sum_delta = torch.zeros_like(self.h[k])
                for i in selected_clients_idx:
                    sum_delta.add_(client_states[i][k].detach().float() - prev[k].detach().float())
                self.h[k].add_(sum_delta, alpha=-(alpha / float(m)))

        # ---- 4) Compute new server model (Algorithm 1 line 264-280) ----
        # θ^t = avg(θ_k^t) - (1/alpha) * h^t
        out = {k: avg[k] - (1.0 / alpha) * self.h[k].to(avg[k].device, dtype=avg[k].dtype) for k in avg}
        return out
