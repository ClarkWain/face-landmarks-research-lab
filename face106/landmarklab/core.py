from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    for override in overrides or []:
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return config


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def wing_loss(prediction: torch.Tensor, target: torch.Tensor, width: float, epsilon: float) -> torch.Tensor:
    distance = (prediction - target).abs()
    constant = width - width * math.log1p(width / epsilon)
    return torch.where(
        distance < width,
        width * torch.log1p(distance / epsilon),
        distance - constant,
    ).mean()


def adaptive_wing_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    omega: float = 14.0,
    epsilon: float = 0.5,
    alpha: float = 2.1,
    theta: float = 0.5,
) -> torch.Tensor:
    """Adaptive Wing Loss for heatmap regression (AWing, Wang et al. 2019).

    Addresses foreground/background imbalance in heatmap training by:
    1. Per-pixel adaptive nonlinearity: large gradient near landmarks, small gradient far away.
    2. Weighted loss map: foreground (near peak) gets higher weight than background.

    Args:
        prediction: predicted heatmap values (B, K, H, W), should be sigmoid(logits) in [0,1]
        target: target Gaussian heatmap (B, K, H, W), values in [0, 1]
        omega: transition point between nonlinear and linear regions
        epsilon: curvature control within nonlinear region
        alpha: adaptiveness parameter (0=standard Wing, >1=more adaptive)
        theta: foreground threshold on target heatmap
    """
    d = (prediction - target).abs()
    d_safe = d.clamp(min=1e-6)  # avoid division by zero in power with negative exponent

    # C1 continuity constant: ensures smooth transition at omega
    a = omega * (1.0 / (1.0 + (omega / epsilon) ** (1.0 - alpha)))
    c = a * math.log(1.0 + (omega / epsilon) ** (1.0 - alpha)) - math.log(1.0 + omega / epsilon)

    # Adaptive Wing Loss element-wise (use d_safe to avoid inf)
    loss_nonlinear = a * torch.log(1.0 + (d_safe / epsilon) ** (1.0 - alpha)) - torch.log(1.0 + d_safe / epsilon)
    loss_linear = d - c
    loss = torch.where(d < omega, loss_nonlinear, loss_linear)

    # Weight map: foreground (target > theta) gets higher weight
    weight_bg = 1.0
    weight_fg = 2.0
    weight = (target.clamp(0, 1) - theta).clamp(min=0) / max(1.0 - theta, 1e-6)
    weight = weight_bg + (weight_fg - weight_bg) * weight

    return (loss * weight).mean()


def build_geometry_edges(num_landmarks: int) -> torch.Tensor:
    if num_landmarks == 98:
        segments = [
            list(range(0, 33)),
            list(range(33, 42)),
            list(range(42, 51)),
            list(range(51, 55)),
            list(range(55, 60)),
            [60, 61, 62, 63, 64, 65, 66, 67, 60],
            [68, 69, 70, 71, 72, 73, 74, 75, 68],
            [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 76],
            [88, 89, 90, 91, 92, 93, 94, 95, 88],
        ]
        edges = []
        for segment in segments:
            edges.extend((segment[index], segment[index + 1]) for index in range(len(segment) - 1))
        return torch.tensor(edges, dtype=torch.long)

    if num_landmarks == 68:
        segments = [
            list(range(0, 17)),
            list(range(17, 22)),
            list(range(22, 27)),
            list(range(27, 31)),
            list(range(31, 36)),
            [36, 37, 38, 39, 40, 41, 36],
            [42, 43, 44, 45, 46, 47, 42],
            [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48],
            [60, 61, 62, 63, 64, 65, 66, 67, 60],
        ]
        edges = []
        for segment in segments:
            edges.extend((segment[index], segment[index + 1]) for index in range(len(segment) - 1))
        return torch.tensor(edges, dtype=torch.long)

    edges = [(index, index + 1) for index in range(num_landmarks - 1)]
    return torch.tensor(edges, dtype=torch.long)


def geometry_loss(prediction: torch.Tensor, target: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    edges = edges.to(prediction.device)
    pred_pairs = prediction[:, edges[:, 0]] - prediction[:, edges[:, 1]]
    target_pairs = target[:, edges[:, 0]] - target[:, edges[:, 1]]
    pred_lengths = pred_pairs.norm(dim=-1)
    target_lengths = target_pairs.norm(dim=-1)
    return F.smooth_l1_loss(pred_lengths, target_lengths)


@torch.no_grad()
def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eye_groups: tuple[list[int], list[int]] | list[list[int]],
) -> dict[str, float]:
    prediction = prediction.view(prediction.size(0), -1, 2)
    target = target.view(target.size(0), -1, 2)
    left_eye = target[:, eye_groups[0]].mean(dim=1)
    right_eye = target[:, eye_groups[1]].mean(dim=1)
    interocular = (left_eye - right_eye).norm(dim=-1).clamp(min=1e-6)
    point_error = (prediction - target).norm(dim=-1) / interocular[:, None]
    image_error = point_error.mean(dim=1)
    metrics = {
        "nme": float(point_error.mean().item()),
        "acc_005": float((point_error < 0.05).float().mean().item() * 100.0),
        "acc_008": float((point_error < 0.08).float().mean().item() * 100.0),
        "acc_010": float((point_error < 0.10).float().mean().item() * 100.0),
        "image_acc_008": float((image_error < 0.08).float().mean().item() * 100.0),
    }
    return metrics


def flatten_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_size_mb(model: torch.nn.Module, bytes_per_weight: int = 1) -> float:
    return count_parameters(model) * bytes_per_weight / (1024 * 1024)


def save_history_plot(history: list[dict[str, float]], path: str | Path) -> None:
    if not history:
        return
    epochs = [item["epoch"] for item in history]
    plt.figure(figsize=(11, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [item["train_loss"] for item in history], label="train_loss")
    plt.plot(epochs, [item["valid_loss"] for item in history], label="valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [item["valid_acc_008"] for item in history], label="valid_acc_008")
    plt.plot(epochs, [item["test_acc_008"] for item in history], label="test_acc_008")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def append_optimization_log(
    log_path: str | Path,
    *,
    stage: str,
    run_name: str,
    note: str,
    summary: dict[str, Any],
) -> None:
    record_lines = [
        f"## {datetime.now():%Y-%m-%d %H:%M:%S} | {stage} | {run_name}",
        f"- 优化点: {note}",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            record_lines.append(f"- {key}: {value:.6f}" if abs(value) < 10 else f"- {key}: {value:.3f}")
        else:
            record_lines.append(f"- {key}: {value}")
    record_lines.append("")
    with Path(log_path).open("a", encoding="utf-8") as file:
        file.write("\n".join(record_lines))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
