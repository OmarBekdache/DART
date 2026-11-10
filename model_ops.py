import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np

from contextlib import nullcontext
from queue import Queue
import threading
from dataclasses import field

from dataclasses import dataclass

from typing import Optional, Callable

from methods import AFA, make_method
from models import (
    MobileNet,
    ResNet18,
    ResNet18_parallel_bn,
    VGG,
    set_parallel_bn_use_aux,
)
from utils import progress_bar
from config import Config

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class Client:
    model: nn.Module
    criterion: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler
    client_loader: torch.utils.data.DataLoader

    loss_fn: Optional[LossFn] = None

    device: Optional[torch.device] = None
    cuda_stream: Optional[torch.cuda.Stream] = None
    task_q: Queue = field(default_factory=Queue)
    done_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    error: Optional[Exception] = None


class ModelOperations:
    def __init__(
        self, cfg: Config, num_classes: int, client_loaders: torch.utils.data.DataLoader
    ) -> None:
        self.cfg = cfg
        self.num_classes = num_classes
        device_idx = self.cfg.gpu_idx
        self.device = (
            torch.device(f"cuda:{device_idx}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.server_model = self._generate_model()
        print("Server model initialized.\n")
        self._move_model(self.server_model, self.device)
        print(f"Server model moved to {self.device}.\n")

        try:
            self.method_obj = make_method(self.cfg)
        except KeyError:
            raise ValueError(f"Model type '{self.cfg.method}' is not supported.")

        self.clients = []

        for i in range(self.cfg.num_clients):
            model = self._generate_model()
            crit = nn.CrossEntropyLoss()
            opt = optim.SGD(
                model.parameters(),
                lr=self.cfg.client_learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.cfg.total_communication_rounds
            )

            self.clients.append(Client(model, crit, opt, sched, client_loaders[i]))

        print("Clients initialized successfully.\n")

        self._copy_state_dict(self.server_model, [c.model for c in self.clients])

        for i in range(self.cfg.num_clients):
            self._move_model(self.clients[i].model, self.device)
            self._move_optimizer(self.clients[i].optimizer, self.device)
        print(f"Client models and optimizers moved to {self.device}.\n")

        for c in self.clients:
            c.device = self.device
            if self.device.type == "cuda":
                c.cuda_stream = torch.cuda.Stream(device=c.device)
            c.done_event.clear()
            c.thread = threading.Thread(
                target=self._client_worker_loop, args=(c,), daemon=True
            )
            c.thread.start()
        print("Client worker threads started.\n")

    def _generate_model(self) -> nn.Module:
        """
        Create and return a model instance based on the configuration.
        """
        n_classes = self.num_classes

        resnet18_ctor = (
            (lambda: ResNet18_parallel_bn(n_classes))
            if self.cfg.method == "FedAFA"
            else (lambda: ResNet18(n_classes))
        )

        model_constructors = {
            "ResNet18": resnet18_ctor,
            "VGG16": lambda: VGG("VGG16", n_classes),
            "MobileNet": lambda: MobileNet(n_classes),
        }

        try:
            return model_constructors[self.cfg.model_type]()
        except KeyError:
            raise ValueError(f"Model type '{self.cfg.model_type}' is not supported.")

    def _move_model(self, model: nn.Module, device: torch.device) -> None:
        """
        Move the model to device.

        Args:
            model (nn.Module): The model to move.
            device (torch.device): The device to move the optimizer to.
        """
        model.to(device)

    def _move_optimizer(self, optimizer: torch.optim, device: torch.device) -> None:
        """
        Move the optimizer to device.

        Args:
            optimizer (torch.optim): The optimizer to move.
            device (torch.device): The device to move the optimizer to.
        """
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device, non_blocking=True)

    def _copy_state_dict(
        self, source_model: nn.Module, target_models: list[nn.Module]
    ) -> None:
        """
        Copy the state_dict (parameters & buffers) from one PyTorch model
        into each model in a sequence of target models.

        Args:
            source_model (nn.Module): The source PyTorch model.
            target_models (Sequence[nn.Module]): The destination models.
        """
        assert isinstance(source_model, nn.Module), (
            "source_model must be a PyTorch model"
        )
        assert all(isinstance(m, nn.Module) for m in target_models), (
            "target_models must be a sequence of PyTorch models"
        )

        state_dict = source_model.state_dict()
        for model in target_models:
            model.load_state_dict(state_dict)

    def _cleanup_after_client_round(self, client: Client) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(client.device)
        client.optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _client_worker_loop(self, client):
        if client.device.type == "cuda":
            torch.cuda.set_device(client.device.index)
        while True:
            task = client.task_q.get()
            if task is None:
                client.task_q.task_done()
                break
            try:
                self._train_on_worker(
                    client, task["local_epochs"], task["show_progress"], task["idx"]
                )
                self._cleanup_after_client_round(client)
            except Exception as exc:
                client.error = exc
            finally:
                client.done_event.set()
                client.task_q.task_done()

    def _train_on_worker(self, client, local_epochs, show_progress, idx):
        model, optimizer = client.model, client.optimizer
        device = client.device
        context = (
            torch.cuda.stream(client.cuda_stream)
            if client.cuda_stream is not None
            else nullcontext()
        )
        with context:
            for epoch in range(local_epochs):
                print(f"Client {idx} — Epoch {epoch}")
                model.train()
                train_loss = correct = total = 0
                for batch_idx, (inputs, targets) in enumerate(client.client_loader):
                    targets = targets.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    if self.cfg.method == "FedAFA":
                        mean = (0.4914, 0.4822, 0.4465)
                        std = (0.2023, 0.1994, 0.2010)
                        transform = transforms.Compose(
                            [
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),
                            ]
                        )
                        main_inputs = torch.stack(
                            [transform(image) for image in inputs]
                        ).to(device)
                        set_parallel_bn_use_aux(model, use_aux=False)
                        main_outputs = model(main_inputs)
                        aux_inputs = torch.stack(
                            [transforms.ToTensor()(image) for image in inputs]
                        ).to(device)
                        aux_inputs = AFA(lam=6.0, per_channel=True)(aux_inputs)
                        aux_inputs = (
                            aux_inputs
                            - aux_inputs.new_tensor(mean)[None, :, None, None]
                        ) / aux_inputs.new_tensor(std)[None, :, None, None]
                        set_parallel_bn_use_aux(model, use_aux=True)
                        outputs = model(aux_inputs)
                        loss = client.loss_fn((main_outputs, outputs), targets)
                    else:
                        if self.cfg.method in ("FedAugMix", "FedPrime"):
                            batch = torch.cat(inputs, 0).to(device, non_blocking=True)
                        else:
                            inputs = inputs.to(device, non_blocking=True)
                            batch = inputs
                        outputs = model(batch)
                        loss = client.loss_fn(
                            outputs, targets, inputs=inputs, model=model
                        )
                    loss.backward()
                    optimizer.step()
                    if show_progress:
                        train_loss += loss.item()
                        predicted = outputs[: targets.size(0)].argmax(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                        progress_bar(
                            batch_idx,
                            len(client.client_loader),
                            f"Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100 * correct / total:.3f}%",
                        )
        if client.cuda_stream is not None:
            torch.cuda.current_stream(device).wait_stream(client.cuda_stream)

    def train_client_models(self, selected_clients_idx: list[int]) -> None:
        """
        Train client models in accordance to the method specified in config.

        Args:
            selected_clients_idx (list[int]): :ist of selected clients participating in current training round.
        """
        selected_models = [self.clients[i].model for i in selected_clients_idx]

        self._copy_state_dict(self.server_model, selected_models)
        print("Selected clients now have updated server model.\n")

        for idx in selected_clients_idx:
            c = self.clients[idx]
            c.done_event.clear()

            ctx = dict(
                client_model=c.model,
                client_criterion=c.criterion,
                mu=self.cfg.mu_fedprox,
                server_model=self.server_model,
                client_idx=idx,
            )

            c.loss_fn = self.method_obj.build_client_loss(**ctx)

            c.task_q.put(
                {
                    "cmd": "train",
                    "local_epochs": self.cfg.local_epochs,
                    "show_progress": bool(self.cfg.progress_bar),
                    "idx": idx,
                }
            )

        for idx in selected_clients_idx:
            self.clients[idx].done_event.wait()
        for idx in selected_clients_idx:
            if self.clients[idx].error is not None:
                raise RuntimeError(f"Client {idx} training failed") from self.clients[
                    idx
                ].error

        for idx in selected_clients_idx:
            self.clients[idx].scheduler.step()

    def aggregate(
        self, selected_clients_idx: list[int], per_client_data_samples: list[int]
    ) -> dict[str, torch.Tensor]:
        client_states = {
            i: self.clients[i].model.state_dict() for i in selected_clients_idx
        }

        ctx = dict(
            server_model=self.server_model,
            selected_clients_idx=selected_clients_idx,
            per_client_data_samples=per_client_data_samples,
            client_states=client_states,
            mu=self.cfg.mu_fedprox,
            local_epochs=self.cfg.local_epochs,
            batch_size=self.cfg.client_batch_size,
            drop_last=False,
        )

        aggregated_weights = self.method_obj.aggregate(**ctx)

        self.server_model.load_state_dict(aggregated_weights)

        return aggregated_weights

    def test(self, test_loader: torch.utils.data.DataLoader) -> float:
        """
        Test model on the test set.

        Args:
            test_loader (torch.utils.data.DataLoader): Loader used to load the test set.
            is_DART (bool): If true, student model is under test. If False, server model is under test.

        Return:
            float: Test accuracy on the test set.
        """
        model = self.server_model
        criterion = nn.CrossEntropyLoss()

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        if self.cfg.method == "FedAFA":
            set_parallel_bn_use_aux(model, use_aux=False)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total += targets.size(0)

                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

                if self.cfg.progress_bar:
                    test_loss += loss.item()
                    progress_bar(
                        batch_idx,
                        len(test_loader),
                        "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                        % (
                            test_loss / (batch_idx + 1),
                            100.0 * correct / total,
                            correct,
                            total,
                        ),
                    )

        acc = 100.0 * correct / total
        return acc

    def evaluate_robustness(
        self, corruption_loaders: list[torch.utils.data.DataLoader]
    ) -> tuple[list, float]:
        """
        Common corruption test.

        Args:
            corruption_loaders (list[torch.utils.data.DataLoader]): Loader used to load the corruption sets.
            is_DART (bool): If true, student model is under test. If False, server model is under test.

        Return:
            list: Accuracy for individual corruptions.
            float: Average robust accuracy on all corruptions.
        """
        corruption_accs = {}
        for corruption_name in sorted(corruption_loaders.keys()):
            loader = corruption_loaders[corruption_name]
            print(f"\nTesting on {corruption_name}")
            acc = self.test(loader)
            corruption_accs[corruption_name] = acc
            print(f"{corruption_name}: {acc:.2f}%")

        mean_acc = sum(corruption_accs.values()) / len(corruption_accs)

        print("\n====== Overall Robustness Results ======")
        for corruption_name in sorted(corruption_accs.keys()):
            print(f"{corruption_name:20s}: {corruption_accs[corruption_name]:6.2f}%")
        print(f"\nMean Corruption Accuracy (mCA): {mean_acc:.2f}%")

        return corruption_accs, mean_acc

    def _freeze_bn_running_stats(self, module: nn.Module):
        for m in module.modules():
            if isinstance(
                m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
            ):
                m.eval()  # stops running_mean / running_var updates
                # Keep the affine parameters trainable.
                if m.affine:
                    if m.weight is not None:
                        m.weight.requires_grad_(True)
                    if m.bias is not None:
                        m.bias.requires_grad_(True)

    def _train_DART(self, teacher_model, optimizer, server_loader, DART_alpha):
        self.server_model.train()
        self._freeze_bn_running_stats(self.server_model)

        teacher_model.eval()

        epoch_loss = 0
        distillation_epoch_loss = 0
        consistency_epoch_loss = 0

        for batch_idx, (inputs, targets) in enumerate(server_loader):
            inputs_all = torch.cat(inputs, 0).to(self.device, non_blocking=True)
            input_clean = inputs[0].to(self.device, non_blocking=True)

            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            outputs_clean, outputs_aug1, outputs_aug2 = torch.split(
                self.server_model(inputs_all), inputs[0].size(0)
            )
            p_clean, p_aug1, p_aug2 = (
                F.softmax(outputs_clean, dim=1),
                F.softmax(outputs_aug1, dim=1),
                F.softmax(outputs_aug2, dim=1),
            )

            with torch.no_grad():
                output_teacher = teacher_model(input_clean)
            p_clean_teacher = F.softmax(output_teacher, dim=1)

            distillation_loss = F.kl_div(
                torch.clamp(p_clean, 1e-7, 1.0).log(),
                torch.clamp(p_clean_teacher, 1e-7, 1.0),
                reduction="batchmean",
            )
            loss = distillation_loss.clone()

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1).log()
            consistency_loss = (
                F.kl_div(p_mixture, p_clean, reduction="batchmean")
                + F.kl_div(p_mixture, p_aug1, reduction="batchmean")
                + F.kl_div(p_mixture, p_aug2, reduction="batchmean")
            ) / 3.0
            loss += DART_alpha * consistency_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            distillation_epoch_loss += distillation_loss.item()
            consistency_epoch_loss += consistency_loss.item()

        average_loss = epoch_loss / len(server_loader)
        average_distillation_loss = distillation_epoch_loss / len(server_loader)
        average_consistency_loss = consistency_epoch_loss / len(server_loader)

        return average_loss, average_distillation_loss, average_consistency_loss

    def _val_DART(self, teacher_model, server_loader, DART_alpha):
        self.server_model.eval()
        self._freeze_bn_running_stats(self.server_model)
        teacher_model.eval()

        epoch_loss = 0
        distillation_epoch_loss = 0
        consistency_epoch_loss = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(server_loader):
                inputs_all = torch.cat(inputs, 0).to(self.device, non_blocking=True)
                input_clean = inputs[0].to(self.device, non_blocking=True)

                targets = targets.to(self.device, non_blocking=True)

                outputs_clean, outputs_aug1, outputs_aug2 = torch.split(
                    self.server_model(inputs_all), inputs[0].size(0)
                )
                p_clean, p_aug1, p_aug2 = (
                    F.softmax(outputs_clean, dim=1),
                    F.softmax(outputs_aug1, dim=1),
                    F.softmax(outputs_aug2, dim=1),
                )

                output_teacher = teacher_model(input_clean)
                p_clean_teacher = F.softmax(output_teacher, dim=1)

                distillation_loss = F.kl_div(
                    torch.clamp(p_clean, 1e-7, 1.0).log(),
                    torch.clamp(p_clean_teacher, 1e-7, 1.0),
                    reduction="batchmean",
                )
                loss = distillation_loss.clone()

                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp(
                    (p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1
                ).log()
                consistency_loss = (
                    F.kl_div(p_mixture, p_clean, reduction="batchmean")
                    + F.kl_div(p_mixture, p_aug1, reduction="batchmean")
                    + F.kl_div(p_mixture, p_aug2, reduction="batchmean")
                ) / 3.0
                loss += DART_alpha * consistency_loss

                epoch_loss += loss.item()
                distillation_epoch_loss += distillation_loss.item()
                consistency_epoch_loss += consistency_loss.item()

        average_loss = epoch_loss / len(server_loader)
        average_distillation_loss = distillation_epoch_loss / len(server_loader)
        average_consistency_loss = consistency_epoch_loss / len(server_loader)

        return average_loss, average_distillation_loss, average_consistency_loss

    def DART(self, round, server_loader, val_loader):
        teacher_model = self._generate_model()
        teacher_model.load_state_dict(self.server_model.state_dict())
        teacher_model.eval()
        print("Teacher model initialized.")
        teacher_model.to(self.device)
        print(f"Teacher model moved to {self.device}.")

        self.server_model.train()
        self._freeze_bn_running_stats(self.server_model)
        print("BN stats of server model are now frozen ahead of DART.")

        DART_optimizer = optim.SGD(
            self.server_model.parameters(),
            lr=self.cfg.server_learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        DART_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            DART_optimizer, T_max=self.cfg.max_DART_epochs
        )

        loss_arr = np.zeros(self.cfg.max_DART_epochs + 1)
        distillation_loss_arr = np.zeros(self.cfg.max_DART_epochs + 1)
        consistency_loss_arr = np.zeros(self.cfg.max_DART_epochs + 1)

        val_loss_arr = np.zeros(self.cfg.max_DART_epochs + 1)
        val_distillation_loss_arr = np.zeros(self.cfg.max_DART_epochs + 1)
        val_consistency_loss_arr = np.zeros(self.cfg.max_DART_epochs + 1)

        best_val_loss = float("inf")
        no_improvement_count = 0
        best_state_dict = None

        end_epoch = 0

        for e in range(self.cfg.max_DART_epochs):
            print(f"Communication Round {round} - DART epoch {e}:")
            average_loss, average_distillation_loss, average_consistency_loss = (
                self._train_DART(
                    teacher_model, DART_optimizer, server_loader, self.cfg.DART_alpha
                )
            )
            print(
                f"Loss = {average_loss}, Distillation Loss = {average_distillation_loss}, and Consistency Loss = {average_consistency_loss}. \n"
            )

            loss_arr[e] = average_loss
            distillation_loss_arr[e] = average_distillation_loss
            consistency_loss_arr[e] = average_consistency_loss

            val_loss, val_distillation_loss, val_consistency_loss = self._val_DART(
                teacher_model, val_loader, self.cfg.DART_alpha
            )
            print(
                f"Validation Loss = {val_loss}, Validation Distillation Loss = {val_distillation_loss}, and Validation Consistency Loss = {val_consistency_loss}. \n"
            )

            val_loss_arr[e] = val_loss
            val_distillation_loss_arr[e] = val_distillation_loss
            val_consistency_loss_arr[e] = val_consistency_loss

            # Check if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.server_model.state_dict().items()
                }
                end_epoch = e
                torch.save(
                    self.server_model.state_dict(),
                    self.cfg.run_name + f"/DART_checkpoint_round{round}.pth",
                )
                print("Validation loss improved.")
            else:
                no_improvement_count += 1

            # Early Stopping Condition
            if no_improvement_count >= self.cfg.DART_patience:
                print(
                    f"Early stopping at epoch {e} due to no improvement for {self.cfg.DART_patience} epochs."
                )
                break

            print(
                f"Validation Loss: {val_loss}, Early Stopping Counter: {no_improvement_count}"
            )

            DART_scheduler.step()

        if best_state_dict is None:
            raise RuntimeError("DART did not produce a finite validation checkpoint")
        self.server_model.load_state_dict(best_state_dict)
        print("Server model is now the best model after DART.")
        self.server_model.train()
        print("Server model's BN parameters are now unfrozen.")

        return (
            loss_arr,
            distillation_loss_arr,
            consistency_loss_arr,
            val_loss_arr,
            val_distillation_loss_arr,
            val_consistency_loss_arr,
            end_epoch,
        )

    def save_server_model(self) -> None:
        """
        Save the server model.
        """
        torch.save(
            self.server_model.state_dict(), self.cfg.run_name + "/checkpoint.pth"
        )

    def shutdown_workers(self):
        for client in self.clients:
            if client.thread is not None and client.thread.is_alive():
                client.task_q.put(None)
        for client in self.clients:
            if client.thread is not None:
                client.thread.join()
            iterator = getattr(client.client_loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
            client.client_loader = None
