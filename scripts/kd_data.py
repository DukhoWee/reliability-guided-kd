from __future__ import annotations


import argparse


import csv


import json


from pathlib import Path


import sys


from typing import Any, Iterable


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from sklearn.model_selection import train_test_split


from torch.utils.data import DataLoader, Dataset


from tqdm import tqdm


from mcdo_utils import (
    CORRUPTIONS,
    Cifar100CIndexedDataset,
    NumpyImageDataset,
    build_cifar_resnet18,
    build_cifar_resnet50,
    classification_metrics_from_probs,
    entropy_from_logits,
    load_cifar100_clean_arrays,
    load_cifar100c_labels,
    load_model_config,
    load_teacher_state_dict,
    make_normalize_transform,
    predict_probs,
    resolve_device,
    seed_everything,
)


class Cifar100CPairedTrainDataset(Dataset):
    """Return (clean_image, noisy_image, label) for fixed base image ids."""

    def __init__(
        self,
        cifar100c_root: Path,
        clean_images_10k: np.ndarray,
        labels_10k: np.ndarray,
        base_indices: np.ndarray,
        corruptions: Iterable[str],
        severities: Iterable[int],
        transform: object | None = None,
        subset: int | None = None,
    ) -> None:
        corruption_list = list(corruptions)
        if not corruption_list:
            raise ValueError("corruptions must not be empty")
        for corruption in corruption_list:
            if corruption not in CORRUPTIONS:
                raise ValueError(f"Unknown corruption: {corruption}")

        severity_list = [int(severity) for severity in severities]
        if not severity_list:
            raise ValueError("severities must not be empty")
        for severity in severity_list:
            if severity < 1 or severity > 5:
                raise ValueError(f"Severity must be in [1, 5], got {severity}")

        self.clean_images_10k = clean_images_10k
        self.labels_10k = np.asarray(labels_10k, dtype=np.int64)
        self.base_indices = np.asarray(base_indices, dtype=np.int64)
        if subset is not None:
            self.base_indices = self.base_indices[:subset]
        self.corruptions = corruption_list
        self.severities = severity_list
        self.transform = transform
        self.corruption_arrays = {
            corruption: np.load(cifar100c_root / f"{corruption}.npy", mmap_mode="r")
            for corruption in self.corruptions
        }

        self._corr_ids = np.zeros(self.base_indices.shape[0], dtype=np.int64)
        self._sev_ids = np.zeros(self.base_indices.shape[0], dtype=np.int64)
        self.resample_pairs(seed=0)

    def resample_pairs(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._corr_ids = rng.integers(
            low=0,
            high=len(self.corruptions),
            size=self.base_indices.shape[0],
            endpoint=False,
            dtype=np.int64,
        )
        self._sev_ids = rng.integers(
            low=0,
            high=len(self.severities),
            size=self.base_indices.shape[0],
            endpoint=False,
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return int(self.base_indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        base_idx = int(self.base_indices[idx])
        corr_id = int(self._corr_ids[idx])
        sev_id = int(self._sev_ids[idx])
        corruption = self.corruptions[corr_id]
        severity = self.severities[sev_id]

        noisy_index = (severity - 1) * 10000 + base_idx
        clean_image = np.asarray(self.clean_images_10k[base_idx]).copy()
        noisy_image = np.asarray(self.corruption_arrays[corruption][noisy_index]).copy()
        label = int(self.labels_10k[base_idx])

        if self.transform is not None:
            clean_tensor = self.transform(clean_image)
            noisy_tensor = self.transform(noisy_image)
        else:
            clean_tensor = torch.from_numpy(np.transpose(clean_image, (2, 0, 1))).float() / 255.0
            noisy_tensor = torch.from_numpy(np.transpose(noisy_image, (2, 0, 1))).float() / 255.0
        return clean_tensor, noisy_tensor, label


class Cifar100CCdfCalibrationDataset(Dataset):
    """Enumerate all corruption/severity variants for fixed base image ids."""

    def __init__(
        self,
        cifar100c_root: Path,
        clean_images_10k: np.ndarray,
        base_indices: np.ndarray,
        corruptions: Iterable[str],
        severities: Iterable[int],
        transform: object | None = None,
    ) -> None:
        corruption_list = list(corruptions)
        if not corruption_list:
            raise ValueError("corruptions must not be empty")
        for corruption in corruption_list:
            if corruption not in CORRUPTIONS:
                raise ValueError(f"Unknown corruption: {corruption}")

        severity_list = [int(severity) for severity in severities]
        if not severity_list:
            raise ValueError("severities must not be empty")
        for severity in severity_list:
            if severity < 1 or severity > 5:
                raise ValueError(f"Severity must be in [1, 5], got {severity}")

        self.clean_images_10k = clean_images_10k
        self.base_indices = np.asarray(base_indices, dtype=np.int64)
        self.corruptions = corruption_list
        self.severities = severity_list
        self.transform = transform
        self.corruption_arrays = {
            corruption: np.load(cifar100c_root / f"{corruption}.npy", mmap_mode="r")
            for corruption in self.corruptions
        }
        self.variants_per_image = len(self.corruptions) * len(self.severities)

    def __len__(self) -> int:
        return int(self.base_indices.shape[0] * self.variants_per_image)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        base_pos = idx // self.variants_per_image
        variant_pos = idx % self.variants_per_image
        corr_id = variant_pos // len(self.severities)
        sev_id = variant_pos % len(self.severities)

        base_idx = int(self.base_indices[base_pos])
        corruption = self.corruptions[corr_id]
        severity = self.severities[sev_id]
        noisy_index = (severity - 1) * 10000 + base_idx

        clean_image = np.asarray(self.clean_images_10k[base_idx]).copy()
        noisy_image = np.asarray(self.corruption_arrays[corruption][noisy_index]).copy()

        if self.transform is not None:
            clean_tensor = self.transform(clean_image)
            noisy_tensor = self.transform(noisy_image)
        else:
            clean_tensor = torch.from_numpy(np.transpose(clean_image, (2, 0, 1))).float() / 255.0
            noisy_tensor = torch.from_numpy(np.transpose(noisy_image, (2, 0, 1))).float() / 255.0
        return clean_tensor, noisy_tensor


def _can_stratify_split(labels: np.ndarray, heldout_size: int) -> bool:
    if heldout_size <= 0 or heldout_size >= labels.shape[0]:
        return False
    _, counts = np.unique(labels, return_counts=True)
    num_classes = counts.shape[0]
    return (
        heldout_size >= num_classes
        and (labels.shape[0] - heldout_size) >= num_classes
        and bool(np.all(counts >= 2))
    )


def split_fixed_image_ids(
    labels_10k: np.ndarray,
    indices: np.ndarray,
    heldout_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    if heldout_size < 0:
        raise ValueError(f"heldout_size must be >= 0, got {heldout_size}")
    if heldout_size == 0:
        return indices, np.empty((0,), dtype=np.int64)
    if heldout_size >= indices.shape[0]:
        raise ValueError(
            f"heldout_size must be smaller than split size, got {heldout_size} >= {indices.shape[0]}"
        )

    split_labels = np.asarray(labels_10k, dtype=np.int64)[indices]
    stratify = split_labels if _can_stratify_split(split_labels, heldout_size) else None
    kept, heldout = train_test_split(
        indices,
        test_size=heldout_size,
        random_state=seed,
        stratify=stratify,
    )
    return np.asarray(kept, dtype=np.int64), np.asarray(heldout, dtype=np.int64)


def create_cdf_train_val_test_image_id_split(
    labels_10k: np.ndarray,
    seed: int,
    cdf_size: int,
    val_size: int,
    test_size: int,
) -> dict[str, np.ndarray]:
    labels_10k = np.asarray(labels_10k, dtype=np.int64)
    if labels_10k.shape[0] != 10000:
        raise ValueError(f"Expected labels_10k with 10000 entries, got {labels_10k.shape}")

    all_indices = np.arange(labels_10k.shape[0], dtype=np.int64)
    non_cdf_idx, cdf_idx = split_fixed_image_ids(
        labels_10k=labels_10k,
        indices=all_indices,
        heldout_size=cdf_size,
        seed=seed,
    )
    non_test_idx, test_idx = split_fixed_image_ids(
        labels_10k=labels_10k,
        indices=non_cdf_idx,
        heldout_size=test_size,
        seed=seed + 1,
    )
    train_idx, val_idx = split_fixed_image_ids(
        labels_10k=labels_10k,
        indices=non_test_idx,
        heldout_size=val_size,
        seed=seed + 2,
    )
    return {
        "cdf": np.asarray(cdf_idx, dtype=np.int64),
        "train": np.asarray(train_idx, dtype=np.int64),
        "val": np.asarray(val_idx, dtype=np.int64),
        "test": np.asarray(test_idx, dtype=np.int64),
    }


def args_to_jsonable_dict(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def load_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict) and "teacher_state_dict" in payload:
        state_dict = payload["teacher_state_dict"]
    elif isinstance(payload, dict) and "student_state_dict" in payload:
        state_dict = payload["student_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint state_dict is not a dictionary")
    return state_dict


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"global_mean": None, "severity_mean": []}

    acc_values = np.asarray([float(row["acc"]) for row in rows], dtype=np.float64)
    nll_values = np.asarray([float(row["nll"]) for row in rows], dtype=np.float64)
    ece_values = np.asarray([float(row["ece"]) for row in rows], dtype=np.float64)
    ent_values = np.asarray([float(row["pred_entropy_mean"]) for row in rows], dtype=np.float64)

    global_mean = {
        "acc": float(acc_values.mean()),
        "nll": float(nll_values.mean()),
        "ece": float(ece_values.mean()),
        "pred_entropy_mean": float(ent_values.mean()),
    }

    severity_rows = []
    for severity in sorted({int(row["severity"]) for row in rows}):
        severity_subset = [row for row in rows if int(row["severity"]) == severity]
        severity_rows.append(
            {
                "severity": severity,
                "acc": float(np.mean([float(row["acc"]) for row in severity_subset])),
                "nll": float(np.mean([float(row["nll"]) for row in severity_subset])),
                "ece": float(np.mean([float(row["ece"]) for row in severity_subset])),
                "pred_entropy_mean": float(
                    np.mean([float(row["pred_entropy_mean"]) for row in severity_subset])
                ),
            }
        )
    return {"global_mean": global_mean, "severity_mean": severity_rows}
