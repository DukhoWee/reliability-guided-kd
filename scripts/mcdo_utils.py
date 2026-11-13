from __future__ import annotations


import json


import pickle


import random


import tarfile


from pathlib import Path


from typing import Dict, Iterable, List, Sequence, Tuple


import numpy as np


import timm


import torch


import torch.nn as nn


import torch.nn.functional as F


from safetensors.torch import load_file as safetensors_load_file


from sklearn.model_selection import train_test_split


from torch.utils.data import Dataset


CORRUPTIONS: List[str] = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model_config(model_dir: Path) -> Dict[str, object]:
    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def build_cifar_resnet50(num_classes: int) -> nn.Module:
    model = timm.create_model("resnet50", pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_cifar_resnet18(num_classes: int) -> nn.Module:
    model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        cleaned[new_key] = value
    return cleaned


def load_teacher_state_dict(model_dir: Path) -> Dict[str, torch.Tensor]:
    safetensors_path = model_dir / "model.safetensors"
    pytorch_bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        state_dict = safetensors_load_file(str(safetensors_path))
    elif pytorch_bin_path.exists():
        raw = torch.load(str(pytorch_bin_path), map_location="cpu")
        state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    else:
        raise FileNotFoundError(
            f"Could not find model.safetensors or pytorch_model.bin in {model_dir}"
        )
    if not isinstance(state_dict, dict):
        raise TypeError("Loaded checkpoint is not a state_dict dictionary")
    return _strip_common_prefixes(state_dict)


class NormalizeNumpyImage:
    def __init__(self, mean: Iterable[float], std: Iterable[float]) -> None:
        mean_arr = np.asarray(list(mean), dtype=np.float32)
        std_arr = np.asarray(list(std), dtype=np.float32)
        self.mean = torch.tensor(mean_arr).view(3, 1, 1)
        self.std = torch.tensor(std_arr).view(3, 1, 1)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
        return (x - self.mean) / self.std


def make_normalize_transform(mean: Iterable[float], std: Iterable[float]) -> NormalizeNumpyImage:
    return NormalizeNumpyImage(mean=mean, std=std)


class NumpyImageDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray | None = None,
        transform: object | None = None,
        subset: int | None = None,
    ) -> None:
        if indices is None:
            indices = np.arange(len(labels), dtype=np.int64)
        if subset is not None:
            indices = indices[:subset]
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_idx = int(self.indices[idx])
        image = self.images[sample_idx]
        label = int(self.labels[sample_idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
        return image, label


def load_cifar100_clean_arrays(cifar100_tar_path: Path) -> Dict[str, np.ndarray]:
    with tarfile.open(cifar100_tar_path, "r:gz") as tf:
        with tf.extractfile("cifar-100-python/train") as train_f:
            train_payload = pickle.load(train_f, encoding="bytes")
        with tf.extractfile("cifar-100-python/test") as test_f:
            test_payload = pickle.load(test_f, encoding="bytes")

    train_data = np.asarray(train_payload[b"data"], dtype=np.uint8)
    test_data = np.asarray(test_payload[b"data"], dtype=np.uint8)
    train_labels = np.asarray(train_payload[b"fine_labels"], dtype=np.int64)
    test_labels = np.asarray(test_payload[b"fine_labels"], dtype=np.int64)

    train_images = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_images = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }


def load_cifar100c_labels(cifar100c_root: Path) -> np.ndarray:
    labels = np.load(cifar100c_root / "labels.npy")
    return np.asarray(labels, dtype=np.int64)


class Cifar100CIndexedDataset(Dataset):
    """Fixed corruption/severity CIFAR-100-C dataset over arbitrary base image indices."""

    def __init__(
        self,
        cifar100c_root: Path,
        corruption: str,
        severity: int,
        labels_10k: np.ndarray,
        base_indices: np.ndarray,
        transform: object | None = None,
        subset: int | None = None,
    ) -> None:
        if corruption not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption: {corruption}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be in [1, 5], got {severity}")
        if labels_10k.shape[0] != 10000:
            raise ValueError(f"labels_10k must have shape [10000], got {labels_10k.shape}")

        file_path = cifar100c_root / f"{corruption}.npy"
        self.data_50k = np.load(file_path, mmap_mode="r")
        self.corruption = corruption
        self.severity = int(severity)
        self.labels_10k = np.asarray(labels_10k, dtype=np.int64)
        self.base_indices = np.asarray(base_indices, dtype=np.int64)
        if subset is not None:
            self.base_indices = self.base_indices[:subset]
        self.transform = transform

    def __len__(self) -> int:
        return int(self.base_indices.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base_idx = int(self.base_indices[idx])
        sample_idx = (self.severity - 1) * 10000 + base_idx
        image = np.asarray(self.data_50k[sample_idx]).copy()
        label = int(self.labels_10k[base_idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
        return image, label


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_probs = []
    all_labels = []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    if probs.ndim != 2:
        raise ValueError(f"Expected probs shape [N, C], got {tuple(probs.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"Expected labels shape [N], got {tuple(labels.shape)}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probs and labels must have the same first dimension")

    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()
    bin_edges = torch.linspace(0.0, 1.0, steps=n_bins + 1)
    ece = torch.tensor(0.0)
    n = probs.shape[0]

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == 0:
            in_bin = confidences.ge(lower) & confidences.le(upper)
        else:
            in_bin = confidences.gt(lower) & confidences.le(upper)
        count = in_bin.sum().item()
        if count == 0:
            continue
        acc_bin = accuracies[in_bin].mean()
        conf_bin = confidences[in_bin].mean()
        ece += (count / n) * torch.abs(acc_bin - conf_bin)
    return float(ece.item())


def classification_metrics_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    ece_bins: int = 15,
) -> Dict[str, float]:
    probs = probs.float()
    labels = labels.long()
    num_samples = probs.shape[0]
    sample_indices = torch.arange(num_samples)

    confidences, predictions = probs.max(dim=1)
    accuracy = predictions.eq(labels).float().mean().item()
    nll = -torch.log(probs[sample_indices, labels].clamp_min(1e-12)).mean().item()
    ece = compute_ece(probs, labels, n_bins=ece_bins)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

    return {
        "acc": float(accuracy),
        "nll": float(nll),
        "ece": float(ece),
        "pred_entropy_mean": float(entropy.mean().item()),
        "confidence_mean": float(confidences.mean().item()),
    }
