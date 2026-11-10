"""Train standard or DART-enhanced FL with the paper's experiment defaults."""

import json
from pathlib import Path
import time

import numpy as np

from config import Config
from dataset import DataModule
from model_ops import ModelOperations
from utils import set_seed


def main():
    cfg = Config()
    set_seed(cfg.seed)
    start = time.time()
    data = DataModule(cfg)
    client_dataset, test_dataset, corrupt_datasets = data.get_client_dataset()
    num_classes = data.get_num_classes(client_dataset)
    server_dataset = val_dataset = None
    if cfg.server_method == "DART":
        client_dataset, server_dataset, val_dataset = data.get_server_dataset(
            client_dataset
        )
    client_datasets, counts = data.split_dataset(client_dataset)
    client_loaders, server_loader, val_loader, test_loader, corrupt_loaders = (
        data.get_loaders(
            client_datasets,
            server_dataset,
            val_dataset,
            test_dataset,
            corrupt_datasets,
        )
    )
    operations = ModelOperations(cfg, num_classes, client_loaders)
    clean_accuracy, robust_accuracy = [], []
    clean_rounds, robust_rounds = [], []
    corruption_accs = {name: [] for name in corrupt_loaders}
    output = Path(cfg.run_name)
    try:
        for round_idx in range(cfg.total_communication_rounds):
            completed = round_idx + 1
            selected = np.random.choice(
                cfg.num_clients, cfg.clients_per_round, replace=False
            )
            print(f"Round {completed}: selected clients {selected.tolist()}")
            operations.train_client_models(selected)
            operations.aggregate(
                selected, per_client_data_samples=counts.sum(axis=0).tolist()
            )
            if cfg.server_method == "DART" and completed % cfg.dart_period == 0:
                losses = operations.DART(round_idx, server_loader, val_loader)
                np.savez(
                    output / f"dart_losses_round{round_idx}.npz",
                    **dict(
                        zip(
                            (
                                "loss",
                                "distill",
                                "consistency",
                                "val_loss",
                                "val_distill",
                                "val_consistency",
                                "end_epoch",
                            ),
                            losses,
                        )
                    ),
                )
            if completed % 25 == 0 or completed == cfg.total_communication_rounds:
                clean_accuracy.append(operations.test(test_loader))
                clean_rounds.append(completed)
            if completed % 50 == 0 or completed == cfg.total_communication_rounds:
                per_corruption, accuracy = operations.evaluate_robustness(
                    corrupt_loaders
                )
                robust_accuracy.append(accuracy)
                robust_rounds.append(completed)
                for name, accuracy in per_corruption.items():
                    corruption_accs[name].append(accuracy)

        operations.save_server_model()
        np.save(output / "clean_acc.npy", clean_accuracy)
        np.save(output / "robust_acc.npy", robust_accuracy)
        np.savez(output / "corr_acc.npz", **corruption_accs)
        np.savez(
            output / "evaluation_rounds.npz", clean=clean_rounds, robust=robust_rounds
        )
        summary = {
            "clean_accuracy": clean_accuracy[-1],
            "robust_accuracy": robust_accuracy[-1],
            "corruption_accuracy": {
                name: values[-1] for name, values in corruption_accs.items()
            },
            "elapsed_seconds": time.time() - start,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        operations.shutdown_workers()


if __name__ == "__main__":
    main()
