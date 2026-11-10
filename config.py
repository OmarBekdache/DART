"""Command-line configuration for the paper's CIFAR-10 experiments."""

import argparse
import json
from pathlib import Path


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="DART: server-side robust federated learning",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("--run_name", required=True, help="New output directory")
        parser.add_argument("--data_root", default="data", help="Dataset directory")
        parser.add_argument("--gpu_idx", type=int, default=0)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--method",
            default="FedProx",
            choices=(
                "FedAVG",
                "FedProx",
                "FedDyn",
                "FedNova",
                "FedAugMix",
                "FedPrime",
                "FedAFA",
            ),
        )
        parser.add_argument(
            "--model_type",
            default="ResNet18",
            choices=(
                "ResNet18",
                "VGG16",
                "MobileNet",
            ),
        )
        parser.add_argument("--client_ds_name", default="CIFAR10", choices=("CIFAR10",))
        parser.add_argument(
            "--server_ds_name", default="CIFAR100", choices=("CIFAR100",)
        )
        parser.add_argument("--server_method", default="DART", choices=("DART", "None"))
        parser.add_argument("--non_iid_dirichlet_alpha", type=float, default=100)
        parser.add_argument("--num_clients", type=int, default=100)
        parser.add_argument("--clients_per_round", type=int, default=10)
        parser.add_argument("--local_epochs", type=int, default=1)
        parser.add_argument("--total_communication_rounds", type=int, default=1000)
        parser.add_argument("--client_batch_size", type=int, default=64)
        parser.add_argument("--client_learning_rate", type=float, default=0.1)
        parser.add_argument("--client_num_workers", type=int, default=4)
        parser.add_argument("--server_batch_size", type=int, default=128)
        parser.add_argument("--server_learning_rate", type=float, default=0.001)
        parser.add_argument("--server_num_workers", type=int, default=8)
        parser.add_argument("--mixture_width", type=int, default=3)
        parser.add_argument("--mixture_depth", type=int, default=-1)
        parser.add_argument("--aug_severity", type=int, default=3)
        parser.add_argument("--all_ops", action="store_true")
        parser.add_argument("--DART_alpha", type=float, default=10)
        parser.add_argument("--max_DART_epochs", type=int, default=200)
        parser.add_argument("--DART_patience", type=int, default=3)
        parser.add_argument("--dart_period", type=int, default=200)
        parser.add_argument(
            "--mu_fedprox",
            type=float,
            default=1.0,
            help="FedProx proximal coefficient / FedDyn regularization",
        )
        parser.add_argument("--progress_bar", action="store_true")
        args = parser.parse_args()
        positive = (
            "num_clients",
            "clients_per_round",
            "local_epochs",
            "total_communication_rounds",
            "client_batch_size",
            "server_batch_size",
            "client_learning_rate",
            "server_learning_rate",
            "non_iid_dirichlet_alpha",
            "mixture_width",
            "max_DART_epochs",
            "DART_patience",
            "dart_period",
        )
        for name in positive:
            if getattr(args, name) <= 0:
                parser.error(f"--{name} must be positive")
        if args.clients_per_round > args.num_clients:
            parser.error("--clients_per_round cannot exceed --num_clients")
        if args.num_clients > 50000:
            parser.error(
                "--num_clients cannot exceed the number of CIFAR-10 training samples"
            )
        if min(args.client_num_workers, args.server_num_workers, args.gpu_idx) < 0:
            parser.error("Worker counts and --gpu_idx must be nonnegative")
        if args.DART_alpha < 0 or args.mu_fedprox < 0:
            parser.error("Loss weights must be nonnegative")
        if args.method == "FedDyn" and args.mu_fedprox == 0:
            parser.error("FedDyn requires --mu_fedprox > 0")
        if args.method == "FedAFA" and args.model_type != "ResNet18":
            parser.error(
                "FedAFA requires --model_type ResNet18 (parallel batch normalization)"
            )
        self.__dict__.update(vars(args))
        # The paper uses the three-view consistency objective throughout.
        self.no_jsd = False
        output = Path(self.run_name)
        if output.exists():
            parser.error(
                f"Output directory already exists: {output}. Choose a new --run_name."
            )
        output.mkdir(parents=True)
        (output / "config.json").write_text(json.dumps(vars(self), indent=2) + "\n")
        print(json.dumps(vars(self), indent=2))
