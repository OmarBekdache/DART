from .fedafa import AFA, FedAFA
from .fedaugmix import FedAugMix
from .fedavg import FedAVG
from .feddyn import FedDyn
from .fednova import FedNova
from .fedprox import FedProx

__all__ = ["AFA", "make_method"]


def make_method(cfg):
    """Instantiate only the requested client method."""
    constructors = {
        "FedAVG": FedAVG,
        "FedProx": FedProx,
        "FedDyn": FedDyn,
        "FedNova": FedNova,
        "FedAugMix": FedAugMix,
        "FedPrime": FedAugMix,
        "FedAFA": FedAFA,
    }
    return constructors[cfg.method](cfg)
