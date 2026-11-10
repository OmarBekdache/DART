# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0. See licenses/Apache-2.0.txt.
# Adapted for DART's dataset wrappers and experiment configuration.

import torch


import augmentations
import numpy as np


def aug(image, preprocess, mixture_width, mixture_depth, aug_severity, no_jsd, all_ops):
    """Perform AugMix augmentations and compute mixture.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.

    Returns:
      mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    if all_ops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([1] * mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(
        self,
        dataset,
        preprocess,
        mixture_width,
        mixture_depth,
        aug_severity,
        no_jsd,
        all_ops,
    ):
        self.dataset = dataset
        self.preprocess = preprocess
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.no_jsd = no_jsd
        self.all_ops = all_ops

        if hasattr(dataset, "targets"):
            self.targets = dataset.targets
        elif hasattr(dataset, "labels"):
            self.targets = dataset.labels

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess), y
        else:
            im_tuple = (
                self.preprocess(x),
                aug(
                    x,
                    self.preprocess,
                    self.mixture_width,
                    self.mixture_depth,
                    self.aug_severity,
                    self.no_jsd,
                    self.all_ops,
                ),
                aug(
                    x,
                    self.preprocess,
                    self.mixture_width,
                    self.mixture_depth,
                    self.aug_severity,
                    self.no_jsd,
                    self.all_ops,
                ),
            )
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)
