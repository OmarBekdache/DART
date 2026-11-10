"""Evaluate saved checkpoints on CIFAR-10, CIFAR-10-C, C-bar, and P."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from dataset import CIFAR10_STATS, CustomDataset
from models import (
    MobileNet,
    ResNet18,
    ResNet18_parallel_bn,
    VGG,
    set_parallel_bn_use_aux,
)


PERTURBATIONS = (
    "gaussian_noise",
    "shot_noise",
    "motion_blur",
    "zoom_blur",
    "spatter",
    "brightness",
    "translate",
    "rotate",
    "tilt",
    "scale",
)


@torch.no_grad()
def accuracy(model, loader, device):
    correct = total = 0
    for images, targets in loader:
        predicted = model(images.to(device)).argmax(1).cpu()
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return 100.0 * correct / total


def corruption_accuracy(model, directory, transform, args):
    labels = np.load(directory / "labels.npy", allow_pickle=False)
    files = sorted(p for p in directory.glob("*.npy") if p.name != "labels.npy")
    if not files:
        raise ValueError(f"No corruption arrays found in {directory}")
    results = {}
    for path in files:
        images = np.load(path, mmap_mode="r", allow_pickle=False)
        if len(images) % len(labels):
            raise ValueError(f"Image/label count mismatch in {path}")
        targets = np.tile(labels, len(images) // len(labels))
        loader = DataLoader(
            CustomDataset(images, targets, transform),
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        results[path.stem] = accuracy(model, loader, args.device)
    return {
        "mean_accuracy_percent": float(np.mean(list(results.values()))),
        "per_corruption_accuracy_percent": results,
    }


class PerturbationDataset(Dataset):
    def __init__(self, path):
        self.images = np.load(path, mmap_mode="r", allow_pickle=False)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # CIFAR-10-P: video, frame, height, width, channel.
        video = np.array(self.images[index], copy=True).transpose(0, 3, 1, 2)
        return torch.from_numpy(video).float() / 255.0


@torch.no_grad()
def perturbation_stability(model, directory, args):
    results = {}
    for name in PERTURBATIONS:
        loader = DataLoader(
            PerturbationDataset(directory / f"{name}.npy"),
            batch_size=args.video_batch_size,
            num_workers=args.workers,
        )
        total = videos = 0
        for video in loader:
            batch, frames = video.shape[:2]
            if frames < 2:
                raise ValueError("CIFAR-10-P videos must have at least two frames")
            # Preserve the preprocessing used by the paper's evaluate_p.py.
            logits = model(video.flatten(0, 1).to(args.device) * 2 - 1)
            predictions = logits.argmax(1).view(batch, frames)
            reference = predictions[:, :1] if "noise" in name else predictions[:, :-1]
            total += predictions[:, 1:].ne(reference).float().mean(1).sum().item()
            videos += batch
        if videos == 0:
            raise ValueError(f"No videos in {name}")
        results[name] = 100.0 * total / videos
    return {
        "mean_flip_probability_percent": float(np.mean(list(results.values()))),
        "per_perturbation_flip_probability_percent": results,
    }


def evaluate_checkpoint(path, args):
    if args.method == "FedAFA":
        model = ResNet18_parallel_bn(10)
        set_parallel_bn_use_aux(model, False)
    else:
        model = {
            "ResNet18": lambda: ResNet18(10),
            "VGG16": lambda: VGG("VGG16", 10),
            "MobileNet": lambda: MobileNet(10),
        }[args.model_type]()
    load_options = (
        {"weights_only": True}
        if "weights_only" in inspect.signature(torch.load).parameters
        else {}
    )
    state = torch.load(path, map_location="cpu", **load_options)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    result = {"checkpoint": str(path)}
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*CIFAR10_STATS)]
    )
    root = Path(args.data_root)
    if any(benchmark in args.benchmarks for benchmark in ("clean", "C", "C-bar")):
        if "clean" in args.benchmarks:
            test = datasets.CIFAR10(
                root=root / "CIFAR10", train=False, download=True, transform=transform
            )
            result["clean_accuracy_percent"] = accuracy(
                model,
                DataLoader(test, batch_size=args.batch_size, num_workers=args.workers),
                args.device,
            )
        for benchmark, directory in (("C", "CIFAR-10-C"), ("C-bar", "CIFAR-10-C-bar")):
            if benchmark in args.benchmarks:
                result[benchmark] = corruption_accuracy(
                    model, root / directory, transform, args
                )
    if "P" in args.benchmarks:
        result["P"] = perturbation_stability(model, root / "CIFAR-10-P", args)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints", nargs="+", type=Path, help="Checkpoints from matched seeds"
    )
    parser.add_argument("--data_root", default="data")
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
        "--model_type", default="ResNet18", choices=("ResNet18", "VGG16", "MobileNet")
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["clean", "C"],
        choices=("clean", "C", "C-bar", "P"),
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--video_batch_size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, help="New JSON file for results")
    args = parser.parse_args()
    if min(args.batch_size, args.video_batch_size) <= 0 or args.workers < 0:
        parser.error("Batch sizes must be positive and workers nonnegative")
    if args.method == "FedAFA" and args.model_type != "ResNet18":
        parser.error("FedAFA requires ResNet18")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results = [evaluate_checkpoint(path, args) for path in args.checkpoints]
    metrics = {}
    for key in results[0]:
        if key == "checkpoint":
            continue
        values = [item[key] for item in results]
        if isinstance(values[0], dict):
            metric = next(name for name in values[0] if name.startswith("mean_"))
            values = [value[metric] for value in values]
        metrics[key] = {
            "mean_percent": float(np.mean(values)),
            "sample_std_percent": float(np.std(values, ddof=1))
            if len(values) > 1
            else None,
        }
    report = {"results": results, "summary": metrics}
    encoded = json.dumps(report, indent=2) + "\n"
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(encoded)


if __name__ == "__main__":
    main()
