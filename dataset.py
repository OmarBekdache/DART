import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, Subset, DataLoader
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from augmix_utils import AugMixDataset
from prime_utils import PrimeDataset

CIFAR10_STATS = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
CIFAR100_STATS = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = np.array(self.data[idx], copy=True)
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


class DataModule:
    def __init__(self, cfg):
        self.cfg = cfg

    def _norm(self, ds_name):
        # Return mean and std for normalization
        if ds_name == "CIFAR10":
            return CIFAR10_STATS
        elif ds_name == "CIFAR100":
            return CIFAR100_STATS
        else:
            raise ValueError(f"Dataset {ds_name} not supported.")

    def _client_transforms(self):
        # Return client data transforms
        mean, std = self._norm(self.cfg.client_ds_name)

        if self.cfg.method in ("FedAVG", "FedProx", "FedDyn", "FedNova"):
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

        elif self.cfg.method in ("FedAFA"):
            transform_train = None

        elif self.cfg.method in ("FedAugMix"):
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            preprocess = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )
            transform_train = (transform, preprocess)

        elif self.cfg.method in ("FedPrime"):
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            preprocess = transforms.Compose([transforms.Normalize(mean, std)])
            transform_train = (transform, preprocess)

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        return transform_train, transform_test

    def _server_transforms(self):
        mean, std = self._norm(
            self.cfg.client_ds_name
        )  # use same normalization as client
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        return transform_train, preprocess

    def _get_corrupted_test_dataset(self, transform_test):
        ds_name = self.cfg.client_ds_name
        if ds_name == "CIFAR10":
            directory = os.path.join(self.cfg.data_root, "CIFAR-10-C")
            labels = np.load(os.path.join(directory, "labels.npy"))
            corrupt_datasets = {}
            for filename in os.listdir(directory):
                if filename.endswith(".npy") and filename != "labels.npy":
                    file_path = os.path.join(directory, filename)
                    data = np.load(file_path)
                    corruption_name = filename.replace(".npy", "")
                    corruption_dataset = CustomDataset(
                        data, labels, transform=transform_test
                    )
                    corrupt_datasets[corruption_name] = corruption_dataset
        else:
            raise ValueError(f"Dataset {ds_name} not supported.")

        return corrupt_datasets

    def _get_subset_class_counts(self, subset, parent_dataset, num_classes):
        all_labels = np.array(parent_dataset.targets)
        labels = all_labels[subset.indices]
        return np.bincount(labels, minlength=num_classes)

    def get_client_dataset(self):
        ds_name = self.cfg.client_ds_name
        transform_train, transform_test = self._client_transforms()

        if ds_name == "CIFAR10":
            if self.cfg.method == "FedAugMix":
                client_dataset = torchvision.datasets.CIFAR10(
                    root=os.path.join(self.cfg.data_root, "CIFAR10"),
                    train=True,
                    download=True,
                    transform=transform_train[0],
                )
                client_dataset = AugMixDataset(
                    client_dataset,
                    transform_train[1],
                    self.cfg.mixture_width,
                    self.cfg.mixture_depth,
                    self.cfg.aug_severity,
                    self.cfg.no_jsd,
                    self.cfg.all_ops,
                )
            elif self.cfg.method == "FedPrime":
                client_dataset = torchvision.datasets.CIFAR10(
                    root=os.path.join(self.cfg.data_root, "CIFAR10"),
                    train=True,
                    download=True,
                    transform=transform_train[0],
                )
                client_dataset = PrimeDataset(
                    client_dataset,
                    transform_train[1],
                    mixture_width=self.cfg.mixture_width,
                    mixture_depth=self.cfg.mixture_depth,
                    no_jsd=self.cfg.no_jsd,
                    max_depth=3,
                )
            elif self.cfg.method == "FedAFA":
                client_dataset = torchvision.datasets.CIFAR10(
                    root=os.path.join(self.cfg.data_root, "CIFAR10"),
                    train=True,
                    download=True,
                    transform=None,
                )
            else:
                client_dataset = torchvision.datasets.CIFAR10(
                    root=os.path.join(self.cfg.data_root, "CIFAR10"),
                    train=True,
                    download=True,
                    transform=transform_train,
                )
            test_dataset = torchvision.datasets.CIFAR10(
                root=os.path.join(self.cfg.data_root, "CIFAR10"),
                train=False,
                download=True,
                transform=transform_test,
            )
            corrupt_datasets = self._get_corrupted_test_dataset(transform_test)
        else:
            raise ValueError(f"Dataset {ds_name} not supported.")

        return client_dataset, test_dataset, corrupt_datasets

    def get_server_dataset(self, client_dataset):
        # CIFAR-100 labels are ignored by the DART objective.
        transform, preprocess = self._server_transforms()
        root = os.path.join(self.cfg.data_root, "CIFAR100")
        train = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform
        )
        val = torchvision.datasets.CIFAR100(root=root, train=False, download=True)

        def wrap(dataset):
            return AugMixDataset(
                dataset,
                preprocess,
                self.cfg.mixture_width,
                self.cfg.mixture_depth,
                self.cfg.aug_severity,
                self.cfg.no_jsd,
                self.cfg.all_ops,
            )

        return client_dataset, wrap(train), wrap(val)

    def get_num_classes(self, dataset):
        if hasattr(dataset, "targets"):
            num_classes = len(np.unique(dataset.targets))
        elif hasattr(dataset, "labels"):
            num_classes = len(np.unique(dataset.labels))
        return num_classes

    def split_dataset(self, client_dataset):
        num_clients = self.cfg.num_clients
        dataset_size = len(client_dataset)

        if self.cfg.non_iid_dirichlet_alpha is None:
            client_sizes = [dataset_size // num_clients] * num_clients
            for i in range(dataset_size % num_clients):
                client_sizes[i] += 1
            client_datasets = random_split(client_dataset, client_sizes)

        else:
            rng = np.random.default_rng(self.cfg.seed)
            if hasattr(client_dataset, "targets"):
                labels = np.array(client_dataset.targets)
            elif hasattr(client_dataset, "labels"):  # fallback for some datasets
                labels = np.array(client_dataset.labels)
            else:
                raise ValueError("Dataset has no attribute 'targets' or 'labels'.")

            while True:
                classes = np.unique(labels)
                client_indices = [[] for _ in range(num_clients)]

                for c in classes:
                    idx_c = np.where(labels == c)[0]
                    rng.shuffle(idx_c)
                    proportions = rng.dirichlet(
                        [self.cfg.non_iid_dirichlet_alpha] * num_clients
                    )
                    proportions = (proportions * len(idx_c)).astype(int)

                    # fix rounding error so total = len(idx_c)
                    diff = len(idx_c) - np.sum(proportions)
                    for i in range(diff):
                        proportions[i % num_clients] += 1

                    start = 0
                    for client_id, count in enumerate(proportions):
                        client_indices[client_id].extend(idx_c[start : start + count])
                        start += count
                client_datasets = [
                    Subset(client_dataset, inds) for inds in client_indices
                ]

                if any(len(ds) == 0 for ds in client_datasets):
                    empties = [
                        i for i, ds in enumerate(client_datasets) if len(ds) == 0
                    ]
                    print(
                        f"Empty client subsets at indices: {empties}. Reconfiguring ..."
                    )
                else:
                    print("Client subset creation successful.")
                    break

        num_classes = self.get_num_classes(client_dataset)

        count_matrix = np.zeros((len(client_datasets), num_classes), dtype=int)
        for i, subset in enumerate(client_datasets):
            count_matrix[i] = self._get_subset_class_counts(
                subset, client_dataset, num_classes
            )
        count_matrix = count_matrix.T
        np.save(self.cfg.run_name + "/data_per_client_count_matrix", count_matrix)

        # Plot heatmap
        plt.figure(figsize=(10, 6))
        im = plt.imshow(
            count_matrix,
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=5000,
            origin="lower",
        )

        cbar = plt.colorbar(im)
        cbar.set_label("Number of samples", fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        plt.xlabel("Client", fontsize=18)
        plt.ylabel("Class", fontsize=18)

        plt.yticks(range(num_classes), fontsize=14)
        plt.xticks(range(len(client_datasets)), fontsize=14)

        plt.tight_layout()
        plt.savefig(
            self.cfg.run_name
            + f"/client_class_distribution_{self.cfg.non_iid_dirichlet_alpha}.pdf",
            format="pdf",
        )
        plt.close()

        return client_datasets, count_matrix

    def get_loaders(
        self,
        client_datasets,
        server_dataset,
        val_dataset,
        test_dataset,
        corrupt_datasets,
    ):
        generator = torch.Generator().manual_seed(self.cfg.seed)
        client_options = dict(
            batch_size=self.cfg.client_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=self.cfg.client_num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.client_num_workers > 0,
        )
        if self.cfg.method == "FedAFA":
            client_options["collate_fn"] = collate_keep_pil
        clients = [DataLoader(dataset, **client_options) for dataset in client_datasets]
        workers = self.cfg.server_num_workers
        server = (
            None
            if server_dataset is None
            else DataLoader(
                server_dataset,
                batch_size=self.cfg.server_batch_size,
                shuffle=True,
                num_workers=workers,
                pin_memory=True,
            )
        )
        val = (
            None
            if val_dataset is None
            else DataLoader(
                val_dataset, batch_size=1000, num_workers=workers, pin_memory=True
            )
        )
        test = DataLoader(
            test_dataset, batch_size=1000, num_workers=workers, pin_memory=True
        )
        corruptions = {
            name: DataLoader(
                dataset, batch_size=2000, num_workers=workers, pin_memory=True
            )
            for name, dataset in corrupt_datasets.items()
        }
        return clients, server, val, test, corruptions


def collate_keep_pil(batch):
    images, labels = zip(*batch)
    return list(images), torch.as_tensor(labels, dtype=torch.long)
