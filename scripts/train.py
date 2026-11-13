from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mcdo_utils import (
    CORRUPTIONS,
    Cifar100CIndexedDataset,
    NumpyImageDataset,
    build_cifar_resnet18,
    build_cifar_resnet50,
    classification_metrics_from_probs,
    load_cifar100_clean_arrays,
    load_cifar100c_labels,
    load_model_config,
    load_teacher_state_dict,
    make_normalize_transform,
    predict_probs,
    resolve_device,
    seed_everything,
)
from kd_data import (
    Cifar100CCdfCalibrationDataset,
    Cifar100CPairedTrainDataset,
    args_to_jsonable_dict,
    create_cdf_train_val_test_image_id_split,
    load_checkpoint_state_dict,
    summarize_rows,
    write_csv,
)


class Cifar100CPairedEvalDataset(Dataset):
    """Return fixed (clean_image, noisy_image, label) pairs for one corruption/severity."""

    def __init__(
        self,
        cifar100c_root: Path,
        clean_images_10k: np.ndarray,
        labels_10k: np.ndarray,
        base_indices: np.ndarray,
        corruption: str,
        severity: int,
        transform: object | None = None,
    ) -> None:
        if corruption not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption: {corruption}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be in [1, 5], got {severity}")
        self.clean_images_10k = clean_images_10k
        self.labels_10k = np.asarray(labels_10k, dtype=np.int64)
        self.base_indices = np.asarray(base_indices, dtype=np.int64)
        self.corruption = corruption
        self.severity = int(severity)
        self.transform = transform
        self.corruption_array = np.load(cifar100c_root / f"{corruption}.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self.base_indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        base_idx = int(self.base_indices[idx])
        noisy_index = (self.severity - 1) * 10000 + base_idx
        clean_image = np.asarray(self.clean_images_10k[base_idx]).copy()
        noisy_image = np.asarray(self.corruption_array[noisy_index]).copy()
        label = int(self.labels_10k[base_idx])
        if self.transform is not None:
            clean_tensor = self.transform(clean_image)
            noisy_tensor = self.transform(noisy_image)
        else:
            clean_tensor = torch.from_numpy(np.transpose(clean_image, (2, 0, 1))).float() / 255.0
            noisy_tensor = torch.from_numpy(np.transpose(noisy_image, (2, 0, 1))).float() / 255.0
        return clean_tensor, noisy_tensor, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-teacher reliability-gated KD on CIFAR-100-C. "
            "Student sees noisy images; a clean teacher and robust/noise teacher "
            "are selected or blended by uncertainty and teacher disagreement."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/edadaltocg/resnet50_cifar100"))
    parser.add_argument(
        "--clean-teacher-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional ResNet50 clean-teacher checkpoint. Omit to use --model-dir weights."
        ),
    )
    parser.add_argument(
        "--noise-teacher-checkpoint",
        type=Path,
        required=True,
        help="ResNet50 robust/noise-teacher checkpoint.",
    )
    parser.add_argument("--student-model-dir", type=Path, default=Path("models/edadaltocg/resnet18_cifar100"))
    parser.add_argument("--cifar100-tar", type=Path, default=Path("cifar-100-python.tar.gz"))
    parser.add_argument("--cifar100c-root", type=Path, default=Path("CIFAR-100-C"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/multi_teacher_gated_kd"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed used only for the fixed CDF/train/val/test split. Defaults to --seed.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cdf-size", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=950)
    parser.add_argument("--test-size", type=int, default=950)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--alpha-ce", type=float, default=1.0)
    parser.add_argument("--alpha-kd", type=float, default=1.0)
    parser.add_argument("--student-init", type=str, default="pretrained", choices=["pretrained", "random"])
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="val_noisy_nll",
        choices=["val_noisy_nll", "val_clean_nll"],
    )
    parser.add_argument("--ece-bins", type=int, default=15)

    parser.add_argument(
        "--gate-signal",
        type=str,
        default="entropy_js",
        choices=["entropy", "msp", "margin", "js", "entropy_js", "msp_js"],
        help="Reliability score used to trigger fallback.",
    )
    parser.add_argument("--alpha-u", type=float, default=1.0, help="Weight for normalized noisy entropy.")
    parser.add_argument("--beta-js", type=float, default=1.0, help="Weight for normalized JS divergence.")
    parser.add_argument("--gamma-disagree", type=float, default=1.0, help="Weight for clean/noise teacher JS.")
    parser.add_argument(
        "--gate-mode",
        type=str,
        default="selective_blend",
        choices=["selective_blend", "soft_blend", "hard_switch", "fixed_blend"],
        help="How high-risk samples use clean-teacher information.",
    )
    parser.add_argument("--fallback-eta", type=float, default=0.0)
    parser.add_argument(
        "--fallback-clean-weight",
        type=float,
        default=1.0,
        help=(
            "Clean-teacher weight for high-risk samples. 0.5 means an even clean/noise blend."
        ),
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.7,
        help="Calibration quantile for failure threshold when --failure-threshold is omitted.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=None,
        help="Explicit failure threshold. Overrides --threshold-quantile.",
    )

    parser.add_argument("--train-corruptions", nargs="+", default=CORRUPTIONS, choices=CORRUPTIONS)
    parser.add_argument(
        "--train-severities",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
    )
    parser.add_argument("--cdf-corruptions", nargs="+", default=CORRUPTIONS, choices=CORRUPTIONS)
    parser.add_argument(
        "--cdf-severities",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
    )
    parser.add_argument("--val-corruption", type=str, default="gaussian_noise", choices=CORRUPTIONS)
    parser.add_argument("--val-severity", type=int, default=3, choices=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--final-eval-corruptions",
        nargs="+",
        default=CORRUPTIONS,
        choices=CORRUPTIONS,
    )
    parser.add_argument(
        "--final-eval-severities",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
    )
    parser.add_argument("--skip-final-full-eval", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.cdf_size <= 0:
        raise ValueError(f"cdf-size must be > 0, got {args.cdf_size}")
    if args.tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {args.tau}")
    if not 0.0 <= args.fallback_eta <= 1.0:
        raise ValueError(f"fallback-eta must be in [0, 1], got {args.fallback_eta}")
    if not 0.0 <= args.fallback_clean_weight <= 1.0:
        raise ValueError(
            f"fallback-clean-weight must be in [0, 1], got {args.fallback_clean_weight}"
        )
    if not 0.0 <= args.threshold_quantile <= 1.0:
        raise ValueError(f"threshold-quantile must be in [0, 1], got {args.threshold_quantile}")


def build_resnet50_teacher(
    model_dir: Path,
    checkpoint: Path | None,
    num_classes: int,
) -> nn.Module:
    teacher = build_cifar_resnet50(num_classes=num_classes)
    state_dict = load_checkpoint_state_dict(checkpoint) if checkpoint is not None else load_teacher_state_dict(model_dir)
    load_result = teacher.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected key mismatch while loading teacher weights. "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def build_resnet18_student(
    student_model_dir: Path,
    num_classes: int,
    student_init: str,
) -> nn.Module:
    student = build_cifar_resnet18(num_classes=num_classes)
    if student_init == "pretrained":
        state_dict = load_teacher_state_dict(student_model_dir)
        load_result = student.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "Unexpected key mismatch while loading student weights. "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
    elif student_init != "random":
        raise ValueError(f"Unsupported student_init: {student_init}")
    return student


def validate_cifar100c_labels(cifar100c_root: Path, clean_labels_10k: np.ndarray) -> None:
    labels_50k = load_cifar100c_labels(cifar100c_root)
    if labels_50k.shape[0] != 50000:
        raise ValueError(f"Expected CIFAR-100-C labels with 50000 entries, got {labels_50k.shape[0]}")
    for severity in range(1, 6):
        start = (severity - 1) * 10000
        end = severity * 10000
        if not np.array_equal(labels_50k[start:end], clean_labels_10k):
            raise ValueError("CIFAR-100-C labels do not match CIFAR-100 clean test labels.")


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)


def js_divergence_probs(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p_safe = p.clamp_min(1e-12)
    q_safe = q.clamp_min(1e-12)
    m = (0.5 * (p_safe + q_safe)).clamp_min(1e-12)
    kl_pm = (p_safe * (torch.log(p_safe) - torch.log(m))).sum(dim=1)
    kl_qm = (q_safe * (torch.log(q_safe) - torch.log(m))).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def reliability_components(
    logits_clean: torch.Tensor,
    logits_noisy: torch.Tensor,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    probs_clean = F.softmax(logits_clean, dim=1)
    probs_noisy = F.softmax(logits_noisy, dim=1)
    entropy_norm = entropy_from_probs(probs_noisy) / math.log(float(num_classes))
    top2 = torch.topk(probs_noisy, k=2, dim=1).values
    msp_uncertainty = 1.0 - top2[:, 0]
    margin_uncertainty = 1.0 - (top2[:, 0] - top2[:, 1])
    js_norm = js_divergence_probs(probs_clean, probs_noisy) / math.log(2.0)
    return {
        "entropy": entropy_norm,
        "msp": msp_uncertainty,
        "margin": margin_uncertainty,
        "js": js_norm,
    }


def reliability_score(
    components: dict[str, torch.Tensor],
    gate_signal: str,
    alpha_u: float,
    beta_js: float,
    gamma_disagree: float,
) -> torch.Tensor:
    if gate_signal == "entropy":
        return components["entropy"]
    if gate_signal == "msp":
        return components["msp"]
    if gate_signal == "margin":
        return components["margin"]
    if gate_signal == "js":
        return components["js"]
    if gate_signal == "entropy_js":
        return alpha_u * components["entropy"] + beta_js * gamma_disagree * components["js"]
    if gate_signal == "msp_js":
        return alpha_u * components["msp"] + beta_js * gamma_disagree * components["js"]
    raise ValueError(f"Unsupported gate signal: {gate_signal}")


def kd_kl_with_probs_per_sample(
    student_logits: torch.Tensor,
    target_probs: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    log_p_s = F.log_softmax(student_logits / tau, dim=1)
    kl = F.kl_div(log_p_s, target_probs, reduction="none").sum(dim=1)
    return kl * (tau * tau)


def summarize_values(values: torch.Tensor | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values.detach().float().cpu().numpy() if isinstance(values, torch.Tensor) else values)
    arr = arr.astype(np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot summarize empty values")
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


@torch.no_grad()
def compute_calibration_scores(
    clean_teacher: nn.Module,
    noise_teacher: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    gate_signal: str,
    alpha_u: float,
    beta_js: float,
    gamma_disagree: float,
) -> dict[str, torch.Tensor]:
    clean_teacher.eval()
    noise_teacher.eval()
    all_scores: list[torch.Tensor] = []
    all_entropy: list[torch.Tensor] = []
    all_msp: list[torch.Tensor] = []
    all_margin: list[torch.Tensor] = []
    all_js: list[torch.Tensor] = []
    progress = tqdm(dataloader, desc="calibrate", leave=False, disable=not sys.stdout.isatty())
    for clean_images, noisy_images in progress:
        clean_images = clean_images.to(device, non_blocking=True)
        noisy_images = noisy_images.to(device, non_blocking=True)
        logits_clean = clean_teacher(clean_images)
        logits_noisy = noise_teacher(noisy_images)
        components = reliability_components(logits_clean, logits_noisy, num_classes)
        score = reliability_score(components, gate_signal, alpha_u, beta_js, gamma_disagree)
        all_scores.append(score.detach().float().cpu())
        all_entropy.append(components["entropy"].detach().float().cpu())
        all_msp.append(components["msp"].detach().float().cpu())
        all_margin.append(components["margin"].detach().float().cpu())
        all_js.append(components["js"].detach().float().cpu())
    if not all_scores:
        raise ValueError("No calibration scores were computed")
    return {
        "score": torch.cat(all_scores, dim=0),
        "entropy": torch.cat(all_entropy, dim=0),
        "msp": torch.cat(all_msp, dim=0),
        "margin": torch.cat(all_margin, dim=0),
        "js": torch.cat(all_js, dim=0),
    }


def calibration_summary(
    scores: dict[str, torch.Tensor],
    threshold: float,
    threshold_source: str,
    threshold_quantile: float,
) -> dict[str, Any]:
    score_values = scores["score"].numpy()
    return {
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "threshold_quantile": float(threshold_quantile),
        "fallback_ratio": float(np.mean(score_values > threshold)),
        "score": summarize_values(scores["score"]),
        "entropy": summarize_values(scores["entropy"]),
        "msp": summarize_values(scores["msp"]),
        "margin": summarize_values(scores["margin"]),
        "js": summarize_values(scores["js"]),
    }


def evaluate_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    ece_bins: int,
) -> dict[str, float]:
    probs, labels = predict_probs(model=model, dataloader=dataloader, device=device)
    metrics = classification_metrics_from_probs(probs, labels, ece_bins=ece_bins)
    metrics["num_samples"] = float(labels.shape[0])
    return metrics


def train_one_epoch(
    clean_teacher: nn.Module,
    noise_teacher: nn.Module,
    student: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    gate_signal: str,
    gate_mode: str,
    threshold: float,
    alpha_u: float,
    beta_js: float,
    gamma_disagree: float,
    fallback_eta: float,
    fallback_clean_weight: float,
    tau: float,
    alpha_ce: float,
    alpha_kd: float,
    max_grad_norm: float | None,
) -> dict[str, float]:
    ce_criterion = nn.CrossEntropyLoss()
    student.train()
    clean_teacher.eval()
    noise_teacher.eval()

    running_total = 0
    running_correct = 0
    fallback_count = 0
    noisy_count = 0
    sum_loss_total = 0.0
    sum_loss_ce = 0.0
    sum_loss_kd = 0.0
    sum_loss_kd_noisy = 0.0
    sum_loss_kd_fallback = 0.0
    sum_gate_score = 0.0
    sum_entropy = 0.0
    sum_js = 0.0

    progress = tqdm(dataloader, desc="train", leave=False, disable=not sys.stdout.isatty())
    for clean_images, noisy_images, labels in progress:
        clean_images = clean_images.to(device, non_blocking=True)
        noisy_images = noisy_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            logits_t_clean = clean_teacher(clean_images)
            logits_t_noisy = noise_teacher(noisy_images)
            components = reliability_components(logits_t_clean, logits_t_noisy, num_classes)
            gate_score = reliability_score(
                components,
                gate_signal,
                alpha_u,
                beta_js,
                gamma_disagree,
            )
            fallback_mask = gate_score > threshold
            if gate_mode == "fixed_blend":
                fallback_mask = torch.ones_like(gate_score, dtype=torch.bool)

            p_clean = F.softmax(logits_t_clean / tau, dim=1)
            p_noisy = F.softmax(logits_t_noisy / tau, dim=1)
            onehot = F.one_hot(labels, num_classes=num_classes).to(dtype=p_clean.dtype)
            if gate_mode == "hard_switch":
                p_teacher_target = torch.where(fallback_mask.unsqueeze(1), p_clean, p_noisy)
            elif gate_mode == "selective_blend":
                p_teacher_blend = (1.0 - fallback_clean_weight) * p_noisy + fallback_clean_weight * p_clean
                p_teacher_target = torch.where(fallback_mask.unsqueeze(1), p_teacher_blend, p_noisy)
            elif gate_mode == "soft_blend":
                blend_weight = (fallback_clean_weight * gate_score.clamp(0.0, 1.0)).unsqueeze(1)
                p_teacher_target = (1.0 - blend_weight) * p_noisy + blend_weight * p_clean
            elif gate_mode == "fixed_blend":
                p_teacher_target = (
                    (1.0 - fallback_clean_weight) * p_noisy
                    + fallback_clean_weight * p_clean
                )
            else:
                raise ValueError(f"Unsupported gate_mode: {gate_mode}")
            p_target = fallback_eta * onehot + (1.0 - fallback_eta) * p_teacher_target

        logits_s_noisy = student(noisy_images)
        loss_ce = ce_criterion(logits_s_noisy, labels)
        kd_per_sample = kd_kl_with_probs_per_sample(logits_s_noisy, p_target, tau)
        loss_kd = kd_per_sample.mean()
        loss = alpha_ce * loss_ce + alpha_kd * loss_kd

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        batch_size = labels.size(0)
        fallback_batch = int(fallback_mask.sum().item())
        noisy_batch = batch_size - fallback_batch
        running_total += batch_size
        running_correct += int((logits_s_noisy.argmax(dim=1) == labels).sum().item())
        fallback_count += fallback_batch
        noisy_count += noisy_batch
        sum_loss_total += float(loss.item()) * batch_size
        sum_loss_ce += float(loss_ce.item()) * batch_size
        sum_loss_kd += float(loss_kd.item()) * batch_size
        if noisy_batch > 0:
            sum_loss_kd_noisy += float(kd_per_sample[~fallback_mask].sum().item())
        if fallback_batch > 0:
            sum_loss_kd_fallback += float(kd_per_sample[fallback_mask].sum().item())
        sum_gate_score += float(gate_score.sum().item())
        sum_entropy += float(components["entropy"].sum().item())
        sum_js += float(components["js"].sum().item())

        progress.set_postfix(
            total=f"{loss.item():.4f}",
            kd=f"{loss_kd.item():.4f}",
            fb=f"{fallback_batch / max(batch_size, 1):.3f}",
        )

    denom = max(running_total, 1)
    return {
        "train_loss_total": sum_loss_total / denom,
        "train_loss_ce": sum_loss_ce / denom,
        "train_loss_kd": sum_loss_kd / denom,
        "train_loss_kd_noisy": sum_loss_kd_noisy / max(noisy_count, 1),
        "train_loss_kd_fallback": sum_loss_kd_fallback / max(fallback_count, 1),
        "train_acc": running_correct / denom,
        "train_fallback_ratio": fallback_count / denom,
        "train_gate_score_mean": sum_gate_score / denom,
        "train_entropy_mean": sum_entropy / denom,
        "train_js_mean": sum_js / denom,
    }


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(y_true).shape[0] < 2:
        return None
    return finite_or_none(float(average_precision_score(y_true, scores)))


@torch.no_grad()
def analyze_teacher_gate_loader(
    clean_teacher: nn.Module,
    noise_teacher: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    gate_signal: str,
    threshold: float,
    alpha_u: float,
    beta_js: float,
    gamma_disagree: float,
) -> dict[str, float | None]:
    clean_teacher.eval()
    noise_teacher.eval()
    all_scores: list[np.ndarray] = []
    all_entropy: list[np.ndarray] = []
    all_msp: list[np.ndarray] = []
    all_margin: list[np.ndarray] = []
    all_js: list[np.ndarray] = []
    all_fallback: list[np.ndarray] = []
    all_teacher_error: list[np.ndarray] = []
    all_disagreement: list[np.ndarray] = []

    for clean_images, noisy_images, labels in dataloader:
        clean_images = clean_images.to(device, non_blocking=True)
        noisy_images = noisy_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits_clean = clean_teacher(clean_images)
        logits_noisy = noise_teacher(noisy_images)
        components = reliability_components(logits_clean, logits_noisy, num_classes)
        score = reliability_score(components, gate_signal, alpha_u, beta_js, gamma_disagree)
        pred_clean = logits_clean.argmax(dim=1)
        pred_noisy = logits_noisy.argmax(dim=1)

        all_scores.append(score.detach().cpu().numpy())
        all_entropy.append(components["entropy"].detach().cpu().numpy())
        all_msp.append(components["msp"].detach().cpu().numpy())
        all_margin.append(components["margin"].detach().cpu().numpy())
        all_js.append(components["js"].detach().cpu().numpy())
        all_fallback.append(score.gt(threshold).detach().cpu().numpy().astype(np.float64))
        all_teacher_error.append(pred_noisy.ne(labels).detach().cpu().numpy().astype(np.float64))
        all_disagreement.append(pred_clean.ne(pred_noisy).detach().cpu().numpy().astype(np.float64))

    scores = np.concatenate(all_scores).astype(np.float64)
    entropy = np.concatenate(all_entropy).astype(np.float64)
    msp = np.concatenate(all_msp).astype(np.float64)
    margin = np.concatenate(all_margin).astype(np.float64)
    js = np.concatenate(all_js).astype(np.float64)
    fallback = np.concatenate(all_fallback).astype(np.float64)
    teacher_error = np.concatenate(all_teacher_error).astype(np.float64)
    disagreement = np.concatenate(all_disagreement).astype(np.float64)

    # ============================================================
    # Bin-wise teacher error rate by gate score
    # ============================================================
    num_bins = 10
    edges = np.quantile(scores, np.linspace(0.0, 1.0, num_bins + 1))
    edges = np.unique(edges)

    bin_rows = []
    for bin_id in range(len(edges) - 1):
        left = edges[bin_id]
        right = edges[bin_id + 1]

        if bin_id < len(edges) - 2:
            mask = (scores >= left) & (scores < right)
        else:
            mask = (scores >= left) & (scores <= right)

        if mask.sum() == 0:
            continue

        bin_rows.append(
            {
                "bin_id": int(bin_id),
                "bin_left": float(left),
                "bin_right": float(right),
                "avg_gate_score": float(scores[mask].mean()),
                "avg_entropy": float(entropy[mask].mean()),
                "avg_js": float(js[mask].mean()),
                "teacher_error_rate": float(teacher_error[mask].mean()),
                "count": int(mask.sum()),
            }
        )

    return {
        "num_samples": float(scores.shape[0]),
        "fallback_ratio": float(fallback.mean()),
        "gate_score_mean": float(scores.mean()),
        "entropy_mean": float(entropy.mean()),
        "msp_mean": float(msp.mean()),
        "margin_mean": float(margin.mean()),
        "js_mean": float(js.mean()),
        "teacher_noisy_error_rate": float(teacher_error.mean()),
        "top1_disagreement_rate": float(disagreement.mean()),
        "teacher_error_auprc": safe_average_precision(teacher_error, scores),
        "teacher_error_auroc": (
            finite_or_none(float(roc_auc_score(teacher_error, scores)))
            if np.unique(teacher_error).shape[0] >= 2
            else None
        ),
        "gate_bin_rows": bin_rows,
        "raw_scores": scores,
        "raw_teacher_error": teacher_error,
        "raw_entropy": entropy,
        "raw_msp": msp,
        "raw_margin": margin,
        "raw_js": js,
    }


def weighted_gate_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    valid_rows = [row for row in rows if row.get(key) is not None]
    total = sum(int(row["num_samples"]) for row in valid_rows)
    if total <= 0:
        return None
    return float(sum(float(row[key]) * int(row["num_samples"]) for row in valid_rows) / total)


def summarize_gate_rows(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)
    summaries = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        summaries.append(
            {
                group_key: key,
                "num_samples": sum(int(row["num_samples"]) for row in group_rows),
                "fallback_ratio": weighted_gate_mean(group_rows, "fallback_ratio"),
                "gate_score_mean": weighted_gate_mean(group_rows, "gate_score_mean"),
                "entropy_mean": weighted_gate_mean(group_rows, "entropy_mean"),
                "msp_mean": weighted_gate_mean(group_rows, "msp_mean"),
                "margin_mean": weighted_gate_mean(group_rows, "margin_mean"),
                "js_mean": weighted_gate_mean(group_rows, "js_mean"),
                "teacher_noisy_error_rate": weighted_gate_mean(group_rows, "teacher_noisy_error_rate"),
                "top1_disagreement_rate": weighted_gate_mean(group_rows, "top1_disagreement_rate"),
                "teacher_error_auprc": weighted_gate_mean(group_rows, "teacher_error_auprc"),
                "teacher_error_auroc": weighted_gate_mean(group_rows, "teacher_error_auroc"),
            }
        )
    return summaries


def write_optional_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_metric: float,
    split: dict[str, np.ndarray],
    model_config: dict[str, Any],
    student_model_config: dict[str, Any],
    args: argparse.Namespace,
    threshold: float,
    calibration: dict[str, Any],
) -> None:
    payload = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "args": args_to_jsonable_dict(args),
        "baseline_type": "multi_teacher_gated_kd",
        "gate_config": {
            "gate_signal": args.gate_signal,
            "gate_mode": args.gate_mode,
            "alpha_u": float(args.alpha_u),
            "beta_js": float(args.beta_js),
            "gamma_disagree": float(args.gamma_disagree),
            "fallback_eta": float(args.fallback_eta),
            "fallback_clean_weight": float(args.fallback_clean_weight),
            "threshold": float(threshold),
            "threshold_quantile": float(args.threshold_quantile),
            "failure_threshold": args.failure_threshold,
        },
        "calibration_summary": calibration,
        "model_config": model_config,
        "clean_teacher_model_config": model_config,
        "noise_teacher_model_config": model_config,
        "student_model_config": student_model_config,
        "student_arch": "resnet18",
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "split_cdf_idx": split["cdf"],
        "split_train_idx": split["train"],
        "split_val_idx": split["val"],
        "split_test_idx": split["test"],
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_model_config(args.model_dir)
    student_model_cfg = load_model_config(args.student_model_dir)
    mean = model_cfg.get("mean", model_cfg.get("pretrained_cfg", {}).get("mean"))
    std = model_cfg.get("std", model_cfg.get("pretrained_cfg", {}).get("std"))
    if mean is None or std is None:
        raise ValueError("Could not infer mean/std from teacher model config")
    num_classes = int(model_cfg.get("num_classes", 100))
    student_num_classes = int(student_model_cfg.get("num_classes", num_classes))
    if student_num_classes != num_classes:
        raise ValueError(
            f"Teacher/student num_classes mismatch: teacher={num_classes}, student={student_num_classes}"
        )

    print(f"[INFO] Device: {device}")
    print("[INFO] Loading CIFAR-100 clean arrays...")
    arrays = load_cifar100_clean_arrays(args.cifar100_tar)
    clean_images_10k = np.asarray(arrays["test_images"])
    clean_labels_10k = np.asarray(arrays["test_labels"], dtype=np.int64)
    validate_cifar100c_labels(args.cifar100c_root, clean_labels_10k)

    split_seed = args.seed if args.split_seed is None else args.split_seed
    split = create_cdf_train_val_test_image_id_split(
        labels_10k=clean_labels_10k,
        seed=split_seed,
        cdf_size=args.cdf_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    if args.subset is not None:
        split = {key: value[: args.subset] if key != "cdf" else value for key, value in split.items()}
    print(
        "[INFO] Split sizes (base image ids): "
        f"cdf={split['cdf'].shape[0]}, train={split['train'].shape[0]}, "
        f"val={split['val'].shape[0]}, test={split['test'].shape[0]}"
    )

    transform = make_normalize_transform(mean=mean, std=std)
    cdf_ds = Cifar100CCdfCalibrationDataset(
        cifar100c_root=args.cifar100c_root,
        clean_images_10k=clean_images_10k,
        base_indices=split["cdf"],
        corruptions=args.cdf_corruptions,
        severities=args.cdf_severities,
        transform=transform,
    )
    train_ds = Cifar100CPairedTrainDataset(
        cifar100c_root=args.cifar100c_root,
        clean_images_10k=clean_images_10k,
        labels_10k=clean_labels_10k,
        base_indices=split["train"],
        corruptions=args.train_corruptions,
        severities=args.train_severities,
        transform=transform,
    )
    val_clean_ds = NumpyImageDataset(clean_images_10k, clean_labels_10k, split["val"], transform)
    val_noisy_ds = Cifar100CIndexedDataset(
        args.cifar100c_root,
        args.val_corruption,
        args.val_severity,
        clean_labels_10k,
        split["val"],
        transform,
    )
    test_clean_ds = NumpyImageDataset(clean_images_10k, clean_labels_10k, split["test"], transform)
    test_noisy_ds = Cifar100CIndexedDataset(
        args.cifar100c_root,
        args.val_corruption,
        args.val_severity,
        clean_labels_10k,
        split["test"],
        transform,
    )

    pin_memory = device.type == "cuda"
    cdf_loader = DataLoader(
        cdf_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_clean_loader = DataLoader(
        val_clean_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_noisy_loader = DataLoader(
        val_noisy_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_clean_loader = DataLoader(
        test_clean_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_noisy_loader = DataLoader(
        test_noisy_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print("[INFO] Building clean teacher, noise teacher, and student...")
    clean_teacher = build_resnet50_teacher(
        model_dir=args.model_dir,
        checkpoint=args.clean_teacher_checkpoint,
        num_classes=num_classes,
    ).to(device)
    noise_teacher = build_resnet50_teacher(
        model_dir=args.model_dir,
        checkpoint=args.noise_teacher_checkpoint,
        num_classes=num_classes,
    ).to(device)
    student = build_resnet18_student(
        student_model_dir=args.student_model_dir,
        num_classes=num_classes,
        student_init=args.student_init,
    )
    student = student.to(device)

    print(
        "[INFO] Calibrating reliability threshold "
        f"({split['cdf'].shape[0]} image ids x {len(args.cdf_corruptions)} corruptions "
        f"x {len(args.cdf_severities)} severities)..."
    )
    score_payload = compute_calibration_scores(
        clean_teacher=clean_teacher,
        noise_teacher=noise_teacher,
        dataloader=cdf_loader,
        device=device,
        num_classes=num_classes,
        gate_signal=args.gate_signal,
        alpha_u=args.alpha_u,
        beta_js=args.beta_js,
        gamma_disagree=args.gamma_disagree,
    )
    if args.failure_threshold is not None:
        threshold = float(args.failure_threshold)
        threshold_source = "explicit"
    else:
        threshold = float(torch.quantile(score_payload["score"], args.threshold_quantile).item())
        threshold_source = "quantile"
    calib_summary = calibration_summary(
        scores=score_payload,
        threshold=threshold,
        threshold_source=threshold_source,
        threshold_quantile=args.threshold_quantile,
    )
    calib_path = args.output_dir / "calibration_scores.npz"
    np.savez_compressed(
        calib_path,
        score=score_payload["score"].numpy(),
        entropy=score_payload["entropy"].numpy(),
        msp=score_payload["msp"].numpy(),
        margin=score_payload["margin"].numpy(),
        js=score_payload["js"].numpy(),
    )
    print(
        "[INFO] Calibration "
        f"threshold={threshold:.6f} source={threshold_source} "
        f"fallback_ratio={calib_summary['fallback_ratio']:.4f}"
    )

    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_metric = float("inf")
    best_ckpt_path = args.output_dir / f"best_by_{args.checkpoint_metric}.pt"
    history: list[dict[str, Any]] = []

    print(
        "[INFO] Start multi-teacher gated KD "
        f"(gate={args.gate_signal}, mode={args.gate_mode}, threshold={threshold:.6f}, "
        f"eta={args.fallback_eta}, "
        f"clean_w={args.fallback_clean_weight}, "
        f"tau={args.tau}, alpha_ce={args.alpha_ce}, alpha_kd={args.alpha_kd})"
    )
    for epoch in range(1, args.epochs + 1):
        train_ds.resample_pairs(seed=args.seed + epoch)
        train_stats = train_one_epoch(
            clean_teacher=clean_teacher,
            noise_teacher=noise_teacher,
            student=student,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            gate_signal=args.gate_signal,
            gate_mode=args.gate_mode,
            threshold=threshold,
            alpha_u=args.alpha_u,
            beta_js=args.beta_js,
            gamma_disagree=args.gamma_disagree,
            fallback_eta=args.fallback_eta,
            fallback_clean_weight=args.fallback_clean_weight,
            tau=args.tau,
            alpha_ce=args.alpha_ce,
            alpha_kd=args.alpha_kd,
            max_grad_norm=args.max_grad_norm,
        )
        val_clean = evaluate_loader(student, val_clean_loader, device, args.ece_bins)
        val_noisy = evaluate_loader(student, val_noisy_loader, device, args.ece_bins)
        scheduler.step()

        checkpoint_metric = (
            float(val_noisy["nll"])
            if args.checkpoint_metric == "val_noisy_nll"
            else float(val_clean["nll"])
        )
        is_best = checkpoint_metric < best_metric
        if is_best:
            best_metric = checkpoint_metric
            save_checkpoint(
                path=best_ckpt_path,
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                split=split,
                model_config=model_cfg,
                student_model_config=student_model_cfg,
                args=args,
                threshold=threshold,
                calibration=calib_summary,
            )

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **train_stats,
            "val_clean_acc": float(val_clean["acc"]),
            "val_clean_nll": float(val_clean["nll"]),
            "val_clean_ece": float(val_clean["ece"]),
            "val_noisy_acc": float(val_noisy["acc"]),
            "val_noisy_nll": float(val_noisy["nll"]),
            "val_noisy_ece": float(val_noisy["ece"]),
            "checkpoint_metric": checkpoint_metric,
        }
        history.append(row)
        print(
            f"[EPOCH {epoch:03d}] train_acc={row['train_acc']:.4f} "
            f"train_total={row['train_loss_total']:.4f} "
            f"fb={row['train_fallback_ratio']:.4f} "
            f"score={row['train_gate_score_mean']:.4f} "
            f"val_noisy_nll={row['val_noisy_nll']:.4f} {'*BEST*' if is_best else ''}"
        )

    if best_ckpt_path.exists():
        best_payload = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        student.load_state_dict(best_payload["student_state_dict"], strict=True)
        print(f"[INFO] Loaded best checkpoint: {best_ckpt_path}")

    final_val_clean = evaluate_loader(student, val_clean_loader, device, args.ece_bins)
    final_val_noisy = evaluate_loader(student, val_noisy_loader, device, args.ece_bins)
    final_test_clean = evaluate_loader(student, test_clean_loader, device, args.ece_bins)
    final_test_noisy = evaluate_loader(student, test_noisy_loader, device, args.ece_bins)

    full_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    gate_bin_rows: list[dict[str, Any]] = []
    all_gate_scores: list[np.ndarray] = []
    all_teacher_errors: list[np.ndarray] = []
    all_gate_entropy: list[np.ndarray] = []
    all_gate_msp: list[np.ndarray] = []
    all_gate_margin: list[np.ndarray] = []
    all_gate_js: list[np.ndarray] = []
    if not args.skip_final_full_eval:
        print("[INFO] Final full test on corruption x severity...")
        for corruption in args.final_eval_corruptions:
            for severity in args.final_eval_severities:
                dataset = Cifar100CIndexedDataset(
                    args.cifar100c_root,
                    corruption,
                    severity,
                    clean_labels_10k,
                    split["test"],
                    transform,
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=pin_memory,
                )
                metrics = evaluate_loader(student, dataloader, device, args.ece_bins)
                full_rows.append(
                    {
                        "corruption": corruption,
                        "severity": int(severity),
                        "num_samples": int(metrics["num_samples"]),
                        "acc": float(metrics["acc"]),
                        "nll": float(metrics["nll"]),
                        "ece": float(metrics["ece"]),
                        "pred_entropy_mean": float(metrics["pred_entropy_mean"]),
                        "confidence_mean": float(metrics["confidence_mean"]),
                    }
                )

                paired_dataset = Cifar100CPairedEvalDataset(
                    cifar100c_root=args.cifar100c_root,
                    clean_images_10k=clean_images_10k,
                    labels_10k=clean_labels_10k,
                    base_indices=split["test"],
                    corruption=corruption,
                    severity=int(severity),
                    transform=transform,
                )
                paired_loader = DataLoader(
                    paired_dataset,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=pin_memory,
                )
                gate_metrics = analyze_teacher_gate_loader(
                    clean_teacher=clean_teacher,
                    noise_teacher=noise_teacher,
                    dataloader=paired_loader,
                    device=device,
                    num_classes=num_classes,
                    gate_signal=args.gate_signal,
                    threshold=threshold,
                    alpha_u=args.alpha_u,
                    beta_js=args.beta_js,
                    gamma_disagree=args.gamma_disagree,
                )
                all_gate_scores.append(gate_metrics["raw_scores"])
                all_teacher_errors.append(gate_metrics["raw_teacher_error"])
                all_gate_entropy.append(gate_metrics["raw_entropy"])
                all_gate_msp.append(gate_metrics["raw_msp"])
                all_gate_margin.append(gate_metrics["raw_margin"])
                all_gate_js.append(gate_metrics["raw_js"])
                gate_rows.append(
                    {
                        "corruption": corruption,
                        "severity": int(severity),
                        "num_samples": int(gate_metrics["num_samples"] or 0),
                        "fallback_ratio": gate_metrics["fallback_ratio"],
                        "gate_score_mean": gate_metrics["gate_score_mean"],
                        "entropy_mean": gate_metrics["entropy_mean"],
                        "msp_mean": gate_metrics["msp_mean"],
                        "margin_mean": gate_metrics["margin_mean"],
                        "js_mean": gate_metrics["js_mean"],
                        "teacher_noisy_error_rate": gate_metrics["teacher_noisy_error_rate"],
                        "top1_disagreement_rate": gate_metrics["top1_disagreement_rate"],
                        "teacher_error_auprc": gate_metrics["teacher_error_auprc"],
                        "teacher_error_auroc": gate_metrics["teacher_error_auroc"],
                    }
                )
                for bin_row in gate_metrics.get("gate_bin_rows", []):
                    gate_bin_rows.append(
                        {
                            "corruption": corruption,
                            "severity": int(severity),
                            **bin_row,
                        }
                    )
                print(
                    f"[C100-C TEST] corr={corruption:>17s} sev={severity} "
                    f"acc={metrics['acc']:.4f} nll={metrics['nll']:.4f} "
                    f"fb={gate_metrics['fallback_ratio']:.4f}"
                )

    full_csv = None
    if full_rows:
        full_csv = args.output_dir / "metrics_test_cifar100c.csv"
        write_csv(full_csv, full_rows)

    gate_csv = None
    gate_by_severity_csv = None
    gate_by_corruption_csv = None
    gate_bin_csv = None
    gate_by_severity = summarize_gate_rows(gate_rows, "severity") if gate_rows else []
    gate_by_corruption = summarize_gate_rows(gate_rows, "corruption") if gate_rows else []
    if gate_rows:
        gate_csv = args.output_dir / "gate_analysis_test_cifar100c.csv"
        gate_by_severity_csv = args.output_dir / "gate_analysis_by_severity.csv"
        gate_by_corruption_csv = args.output_dir / "gate_analysis_by_corruption.csv"
        write_optional_csv(gate_csv, gate_rows)
        write_optional_csv(gate_by_severity_csv, gate_by_severity)
        write_optional_csv(gate_by_corruption_csv, gate_by_corruption)

    if gate_bin_rows:
        gate_bin_csv = args.output_dir / "gate_score_bin_teacher_error_rate_by_corruption_severity.csv"
        write_optional_csv(gate_bin_csv, gate_bin_rows)

    if all_gate_scores:
        scores_all = np.concatenate(all_gate_scores).astype(np.float64)
        teacher_error_all = np.concatenate(all_teacher_errors).astype(np.float64)
        entropy_all = np.concatenate(all_gate_entropy).astype(np.float64)
        msp_all = np.concatenate(all_gate_msp).astype(np.float64)
        margin_all = np.concatenate(all_gate_margin).astype(np.float64)
        js_all = np.concatenate(all_gate_js).astype(np.float64)

        np.savez_compressed(
            args.output_dir / "gate_score_teacher_error_raw.npz",
            scores=scores_all,
            teacher_error=teacher_error_all,
            entropy=entropy_all,
            msp=msp_all,
            margin=margin_all,
            js=js_all,
        )

        num_bins = 10
        edges = np.quantile(scores_all, np.linspace(0.0, 1.0, num_bins + 1))
        edges = np.unique(edges)

        overall_bin_rows = []
        for bin_id in range(len(edges) - 1):
            left = edges[bin_id]
            right = edges[bin_id + 1]

            if bin_id < len(edges) - 2:
                mask = (scores_all >= left) & (scores_all < right)
            else:
                mask = (scores_all >= left) & (scores_all <= right)

            if mask.sum() == 0:
                continue

            overall_bin_rows.append(
                {
                    "bin_id": int(bin_id),
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "avg_gate_score": float(scores_all[mask].mean()),
                    "avg_entropy": float(entropy_all[mask].mean()),
                    "avg_msp": float(msp_all[mask].mean()),
                    "avg_margin": float(margin_all[mask].mean()),
                    "avg_js": float(js_all[mask].mean()),
                    "teacher_error_rate": float(teacher_error_all[mask].mean()),
                    "count": int(mask.sum()),
                }
            )

        overall_gate_bin_csv = args.output_dir / "gate_score_bin_teacher_error_rate_overall.csv"
        write_optional_csv(overall_gate_bin_csv, overall_bin_rows)

    split_path = args.output_dir / "split_indices.npz"
    np.savez_compressed(
        split_path,
        cdf_idx=split["cdf"],
        train_idx=split["train"],
        val_idx=split["val"],
        test_idx=split["test"],
    )

    summary = {
        "config": args_to_jsonable_dict(args),
        "baseline_type": "multi_teacher_gated_kd",
        "gate_config": {
            "gate_signal": args.gate_signal,
            "gate_mode": args.gate_mode,
            "alpha_u": float(args.alpha_u),
            "beta_js": float(args.beta_js),
            "gamma_disagree": float(args.gamma_disagree),
            "fallback_eta": float(args.fallback_eta),
            "fallback_clean_weight": float(args.fallback_clean_weight),
            "threshold": float(threshold),
            "threshold_quantile": float(args.threshold_quantile),
            "failure_threshold": args.failure_threshold,
        },
        "calibration_summary": calib_summary,
        "best_checkpoint": str(best_ckpt_path),
        "best_metric": best_metric,
        "split_sizes": {key: int(value.shape[0]) for key, value in split.items()},
        "final_val_clean": final_val_clean,
        "final_val_noisy": {
            "corruption": args.val_corruption,
            "severity": args.val_severity,
            **final_val_noisy,
        },
        "final_test_clean": final_test_clean,
        "final_test_noisy": {
            "corruption": args.val_corruption,
            "severity": args.val_severity,
            **final_test_noisy,
        },
        "full_cifar100c_test": summarize_rows(full_rows),
        "gate_analysis": {
            "by_corruption_severity": gate_rows,
            "by_severity": gate_by_severity,
            "by_corruption": gate_by_corruption,
        },
        "outputs": {
            "split_indices": str(split_path),
            "calibration_scores": str(calib_path),
            "full_cifar100c_test_csv": str(full_csv) if full_csv is not None else None,
            "gate_analysis_csv": str(gate_csv) if gate_csv is not None else None,
            "gate_analysis_by_severity_csv": (
                str(gate_by_severity_csv) if gate_by_severity_csv is not None else None
            ),
            "gate_analysis_by_corruption_csv": (
                str(gate_by_corruption_csv) if gate_by_corruption_csv is not None else None
            ),
        },
        "history": history,
    }
    summary_path = args.output_dir / "train_history.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved split indices: {split_path}")
    print(f"[INFO] Saved calibration scores: {calib_path}")
    if full_csv is not None:
        print(f"[INFO] Saved full CIFAR-100-C test metrics: {full_csv}")
    if gate_csv is not None:
        print(f"[INFO] Saved gate analysis: {gate_csv}")
    print(f"[INFO] Saved train summary: {summary_path}")
    print(
        "[INFO] Final key metrics "
        f"test_clean_acc={final_test_clean['acc']:.4f} "
        f"test_noisy_acc={final_test_noisy['acc']:.4f}"
    )


if __name__ == "__main__":
    main()
