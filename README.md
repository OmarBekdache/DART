# DART

**A Server-side Plug-in for Resource-efficient Robust Federated Learning**  
Omar Bekdache and Naresh Shanbhag · University of Illinois Urbana-Champaign  
CVPR Workshops 2026 · FedVision

[Paper](https://openaccess.thecvf.com/content/CVPR2026W/FedVision/html/Bekdache_DART_A_Server-side_Plug-in_for_Resource-efficient_Robust_Federated_Learning_CVPRW_2026_paper.html)
· [PDF](https://openaccess.thecvf.com/content/CVPR2026W/FedVision/papers/Bekdache_DART_A_Server-side_Plug-in_for_Resource-efficient_Robust_Federated_Learning_CVPRW_2026_paper.pdf)
· [Citation](#citation)

DART improves robustness to common image corruptions through server-side
training on a public proxy dataset. Clients perform their usual local training.
On the server, a frozen copy of the aggregated model supplies distillation
targets while a consistency objective aligns predictions across augmented views.
Private client samples and server dataset labels are not used by the DART loss.

This release contains the CIFAR-10 client / CIFAR-100 server experiments:
FedAvg, FedProx, FedDyn, and FedNova with and without DART; FedAugMix, FedPrime,
and FedAFA baselines; ResNet-18, VGG-16, and MobileNet architectures; and
CIFAR-10-C, CIFAR-10-C-bar, and CIFAR-10-P evaluation. Appendix experiments with
other server datasets and hardware energy measurements are outside this release.

## Installation

The saved research environment used **Python 3.9, PyTorch 1.12.1, and
torchvision 0.13.1**. The dependency list pins the core versions from that
environment.

```bash
git clone https://github.com/OmarBekdache/DART.git
cd DART
conda create -n dart python=3.9 -y
conda activate dart
python -m pip install -r requirements.txt
```

Full experiments require substantial memory: this implementation keeps a model
and optimizer for every client and trains participating clients concurrently.
A CUDA GPU is recommended. Set `CUDA_VISIBLE_DEVICES` to select a GPU; CPU
execution is also supported for small checks. Reduce `--client_num_workers` and
`--server_num_workers` on machines with limited process or memory capacity.

## Data

CIFAR-10 and CIFAR-100 are downloaded automatically by torchvision. Obtain the
evaluation arrays from the benchmark authors:

- [CIFAR-10-C and CIFAR-10-P](https://github.com/hendrycks/robustness).
- [CIFAR-10-C-bar](https://github.com/facebookresearch/augmentation-corruption).

Use `--data_root` to point to a directory with this layout:

```text
data/
├── CIFAR10/                 # torchvision download directory
├── CIFAR100/                # torchvision download directory
├── CIFAR-10-C/
│   ├── labels.npy
│   └── <corruption>.npy
├── CIFAR-10-C-bar/          # needed only for C-bar evaluation
│   ├── labels.npy
│   └── <corruption>.npy
└── CIFAR-10-P/              # needed only for P evaluation
    └── <perturbation>.npy
```

CIFAR-10-C must be present for training-time reporting. Corruption arrays have
shape `(N, 32, 32, 3)`; perturbation arrays have shape
`(videos, frames, 32, 32, 3)`. Evaluation uses the severities provided in the
arrays. These benchmark samples are used only for reporting.

## Training

Run FedProx with DART using the paper's main settings:

```bash
python main.py --data_root data \
  --run_name results/DART/FedProx/alpha100/seed0 \
  --method FedProx --seed 0 --non_iid_dirichlet_alpha 100
```

Run the corresponding standard FL baseline:

```bash
python main.py --data_root data \
  --run_name results/None/FedProx/alpha100/seed0 \
  --method FedProx --server_method None --seed 0 \
  --non_iid_dirichlet_alpha 100
```

`--method` accepts `FedAVG`, `FedProx`, `FedDyn`, `FedNova`, `FedAugMix`,
`FedPrime`, or `FedAFA`. Select `--model_type VGG16` or `MobileNet` for additional
architectures; FedAFA requires ResNet-18 with parallel batch normalization.
Use `--server_method None` for the robust client baselines.

| Setting | Default |
| --- | --- |
| Clients / clients per round | 100 / 10 |
| Global rounds / local epochs | 1,000 / 1 |
| Client / server learning rate | 0.1 / 0.001 |
| Client / server batch size | 64 / 128 |
| DART period / maximum epochs | 200 rounds / 200 epochs |
| DART consistency weight / early-stopping patience | 10 / 3 |
| FedProx or FedDyn coefficient (`--mu_fedprox`) | 1 |

The server uses CIFAR-100 training images for DART and CIFAR-100 test images for
early stopping; their labels are ignored. DART freezes batch-normalization
running statistics and restores the checkpoint with the lowest server validation
loss. Client optimizers and learning-rate schedules retain the research code's
behavior.

For the complete main ResNet-18 comparison across Dirichlet parameters
`100, 10, 1, 0.1`:

```bash
bash scripts/reproduce.sh data results
```

This launches **220 full training runs sequentially**. Use the individual command
above to run a single experiment. Existing run directories are never overwritten.

Each run saves its configuration, client class counts, final `checkpoint.pth`,
accuracy histories, DART loss histories and checkpoints, and `summary.json`.
Clean accuracy is reported every 25 rounds and corruption accuracy every 50
rounds, with an additional evaluation at the final round for shorter runs.
`evaluation_rounds.npz` records the reporting rounds. See `python main.py --help`
for all options.

## Evaluation

Evaluate a checkpoint on clean CIFAR-10 and all three robustness benchmarks:

```bash
python evaluate.py results/DART/FedProx/alpha100/seed0/checkpoint.pth \
  --data_root data --benchmarks clean C C-bar P \
  --output results/evaluation.json
```

To report the mean and sample standard deviation across matched seeds:

```bash
python evaluate.py results/DART/FedProx/alpha100/seed*/checkpoint.pth \
  --data_root data --benchmarks clean C C-bar \
  --output results/evaluation_seeds.json
```

Match `--model_type` to the trained architecture and pass `--method FedAFA` for
FedAFA checkpoints. Use `--device cpu` for CPU evaluation. Only the requested
benchmarks need to be installed. With the original PyTorch release, evaluate
only checkpoints from a trusted source.

Clean and corruption accuracies are percentages. Corruption accuracy averages
the per-corruption accuracies. CIFAR-10-P reports mean flip probability as a
percentage: noise sequences compare each frame to the first frame; other
sequences compare adjacent frames. The P evaluator preserves the original
`2x - 1` input normalization; clean/C/C-bar evaluation uses CIFAR-10 statistics.

The release preserves the DART loss and the included client-method formulas.
Paths, command-line defaults, worker handling, and evaluation packaging were
cleaned up for portability. Concurrent client training and data-loader scheduling
can affect random-number ordering; bit-for-bit reproduction is not guaranteed.

## Code

| File | Purpose |
| --- | --- |
| `main.py`, `config.py` | Training loop and experiment configuration |
| `model_ops.py` | Client training, aggregation, and server-side DART |
| `dataset.py` | Data loading and Dirichlet client partitions |
| `methods/`, `models/` | Client methods and model architectures |
| `augmentations.py`, `augmix_utils.py`, `prime_utils.py` | Augmentation operators |
| `evaluate.py` | Checkpoint evaluation and seed summaries |

Datasets, trained weights, logs, and generated results are excluded from Git.
Third-party attributions and license texts are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`. No new license
has been assigned to the original DART code.

## Citation

```bibtex
@InProceedings{Bekdache_2026_CVPR,
  author    = {Bekdache, Omar and Shanbhag, Naresh},
  title     = {DART: A Server-side Plug-in for Resource-efficient Robust Federated Learning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2026},
  pages     = {3301--3311}
}
```
