import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from typing import Callable, Dict

import torch
import math

@torch.no_grad()
def fourier_basis_noise(batch_size, height, width, device, per_channel=True):
    y = torch.arange(height, device=device).float()
    x = torch.arange(width,  device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    base = torch.stack([xx, yy], dim=0).unsqueeze(0).unsqueeze(0)

    M = max(height, width)
    if per_channel:
        f = torch.randint(1, M+1, (batch_size, 3, 1, 1), device=device).float()
        omega = torch.rand(batch_size, 3, 1, 1, device=device) * math.pi
    else:
        f = torch.randint(1, M+1, (batch_size, 1, 1, 1), device=device).float()
        omega = torch.rand(batch_size, 1, 1, 1, device=device) * math.pi

    cosw, sinw = torch.cos(omega), torch.sin(omega)
    dir_vec = torch.stack([cosw, sinw], dim=2)
    proj = (dir_vec * base).sum(dim=2)

    wave = torch.sin(2*math.pi * f * proj - math.pi/4)
    wv = wave.view(wave.shape[0], wave.shape[1], -1)
    wv = F.normalize(wv, p=2, dim=-1)
    return wv.view_as(wave)

class AFA:
    def __init__(self, lam=6.0, per_channel=True):
        self.lam = lam
        self.per_channel = per_channel
    @torch.no_grad()
    def __call__(self, imgs):
        B, C, H, W = imgs.shape
        dev = imgs.device
        waves = fourier_basis_noise(B, H, W, dev, self.per_channel)
        sigma = torch.distributions.Exponential(1.0/self.lam).sample((B, C)).to(dev).view(B,C,1,1)
        aug = (imgs + sigma * waves).clamp(0,1)
        return aug

class FedAFA:
    def __init__(self, cfg, **_):
        self.cfg = cfg
    
    def build_client_loss(self, *, client_criterion, **_):
        def loss_fn(outputs, targets, **_):
            loss_main = client_criterion(outputs[0], targets)
            loss_aux = client_criterion(outputs[1], targets)
            return 0.5 * (loss_main + loss_aux)
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