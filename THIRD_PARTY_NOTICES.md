# Third-party acknowledgments

- `augmentations.py` and `augmix_utils.py` contain augmentation code adapted from
  [AugMix](https://github.com/google-research/augmix), Copyright 2019 Google LLC,
  under the [Apache License 2.0](licenses/Apache-2.0.txt).
- `prime_utils.py` includes primitives adapted from
  [PRIME](https://github.com/amodas/PRIME-augmentations), under the
  [Apache License 2.0](licenses/Apache-2.0.txt). The local dataset wrapper integrates
  these primitives with the federated training code.
- `models/resnet.py`, `models/vgg.py`, `models/mobilenet.py`, and the progress-bar
  utility in `utils.py` are adapted from
  [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), Copyright (c) 2017
  liukuang, under the [MIT License](licenses/pytorch-cifar-MIT.txt).
  `models/resnet_parallel_bn.py` adapts that ResNet architecture to parallel batch
  normalization. `utils.py` uses a portable terminal-size fallback.

Upstream license texts and existing copyright notices are retained. These
third-party licenses apply to their respective components; they do not assign a
new license to the original DART code.
