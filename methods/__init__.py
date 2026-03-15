from .fedavg import FedAVG
from .fedprox import FedProx
from .fedadam import FedAdam
from .fednova import FedNova
from .feddyn import FedDyn

from .fedafa import AFA, FedAFA

from .fedaugmix import FedAugMix

def make_methods(cfg):
    return {
        "FedAVG":  FedAVG(cfg),
        "FedProx": FedProx(cfg),
        "FedAdam": FedAdam(cfg),
        "FedNova": FedNova(cfg),
        "FedDyn": FedDyn(cfg),
        
        "FedRandAugment": FedAVG(cfg),
        "FedAugMix": FedAugMix(cfg),
        "FedPrime": FedAugMix(cfg),
        "FedAFA": FedAFA(cfg),
    }