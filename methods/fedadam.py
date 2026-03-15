import torch
import torch.nn as nn

class FedAdam:
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.server_lr = getattr(cfg, "fed_adam_server_lr", 1e-2)
        print(self.server_lr)
        self.beta1 = getattr(cfg, "beta1", 0.9)
        print(self.beta1)
        self.beta2 = getattr(cfg, "beta2", 0.999)
        print(self.beta2)
        self.eps = getattr(cfg, "eps", 1e-8)
        self.bias_correction = getattr(cfg, "bias_correction", True)

        self.m = None  # first moment buffers (state_dict-shaped)
        self.v = None  # second moment buffers (state_dict-shaped)
        self.t = 0     # step counter

    def build_client_loss(self, *, client_criterion, **_):
        def loss_fn(outputs, targets, **_):
            return client_criterion(outputs, targets)
        return loss_fn

    @torch.no_grad()
    def _init_moments_like(self, state_dict):
        self.m, self.v = {}, {}
        for k, v in state_dict.items():
            # Only track floating-point tensors
            if torch.is_floating_point(v):
                self.m[k] = torch.zeros_like(v, dtype=torch.float, device=v.device)
                self.v[k] = torch.zeros_like(v, dtype=torch.float, device=v.device)
            else:
                self.m[k] = None
                self.v[k] = None

    @torch.no_grad()
    def aggregate(self, *, server_model, selected_clients_idx, per_client_data_samples, client_states, **_):
        sdc = server_model.state_dict()
        if self.m is None or self.v is None:
            self._init_moments_like(sdc)

        total = sum(per_client_data_samples[i] for i in selected_clients_idx)
        # Accumulate gradient-like delta g_t
        g = {k: torch.zeros_like(v, dtype=torch.float, device=v.device) if torch.is_floating_point(v) else None
             for k, v in sdc.items()}

        for i in selected_clients_idx:
            w_i = per_client_data_samples[i] / total
            csd = client_states[i]  # client's state_dict (model params)
            for k, sv in sdc.items():
                if g[k] is None or (not torch.is_floating_point(sv)):
                    continue
                # Delta (server - client): this is the FedOpt "gradient" direction
                g[k].add_(sv - csd[k], alpha=w_i)

        self.t += 1
        beta1, beta2, eps, lr = self.beta1, self.beta2, self.eps, self.server_lr

        out = {}
        for k, w_t in sdc.items():
            if not torch.is_floating_point(w_t):
                # Non-float buffers (e.g., BN counters) are copied directly
                out[k] = w_t.clone()
                continue

            # Moments
            m = self.m[k]
            v = self.v[k]
            gt = g[k]

            # If, for any reason, a param wasn't tracked as float
            if gt is None:
                out[k] = w_t.clone()
                continue

            # m_t, v_t updates
            m.mul_(beta1).add_(gt, alpha=1 - beta1)
            # v uses elementwise square of g_t
            v.mul_(beta2).addcmul_(gt, gt, value=(1 - beta2))

            if self.bias_correction:
                # Bias correction (standard Adam)
                m_hat = m / (1 - beta1 ** self.t)
                v_hat = v / (1 - beta2 ** self.t)
            else:
                m_hat, v_hat = m, v

            # Parameter update
            step = lr * m_hat / (v_hat.sqrt().add_(eps))
            out[k] = w_t - step

        return out
