from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from landmarklab.core import (
    adaptive_wing_loss,
    append_optimization_log,
    build_geometry_edges,
    compute_metrics,
    count_parameters,
    ensure_dir,
    flatten_metrics,
    geometry_loss,
    load_config,
    parameter_size_mb,
    save_history_plot,
    set_seed,
    wing_loss,
    write_json,
)
from landmarklab.data import create_dataloaders
from landmarklab.model import LandmarkTranslationWrapper, build_model


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)


    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            value.copy_(value * self.decay + model_state[key] * (1.0 - self.decay))


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return (images * 0.5 + 0.5).clamp(0.0, 1.0)


def render_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    path: Path,
    max_samples: int,
) -> None:
    model.eval()
    rows = 3
    columns = max(1, (max_samples + rows - 1) // rows)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = axes.reshape(rows, columns)
    plotted = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            target = batch["landmarks"]
            prediction = model(images).cpu()
            images = denormalize(batch["image"]).cpu()
            batch_size = images.size(0)
            for index in range(batch_size):
                axis = axes[plotted // columns, plotted % columns]
                image = images[index].permute(1, 2, 0).numpy()
                axis.imshow(image)
                target_points = target[index].numpy() * images.shape[-1]
                prediction_points = prediction[index].numpy() * images.shape[-1]
                axis.scatter(target_points[:, 0], target_points[:, 1], c="lime", s=24, label="gt")
                axis.scatter(prediction_points[:, 0], prediction_points[:, 1], c="red", s=24, marker="x", label="pred")
                axis.set_axis_off()
                plotted += 1
                if plotted == 1:
                    axis.legend(loc="lower right")
                if plotted >= max_samples:
                    break
            if plotted >= max_samples:
                break
    for index in range(plotted, rows * columns):
        axes[index // columns, index % columns].set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def estimate_mean_shape(dataset: torch.utils.data.Dataset, max_samples: int = 512) -> torch.Tensor:
    total = None
    count = min(len(dataset), max_samples)
    for index in range(count):
        landmarks = dataset[index]["landmarks"]
        total = landmarks if total is None else total + landmarks
    if total is None:
        raise RuntimeError("Unable to estimate mean shape from an empty dataset")
    return total / count


def load_compatible_state_dict(
    model: nn.Module,
    source_state: dict[str, torch.Tensor],
    exclude_keys: list[str] | None = None,
) -> int:
    target_state = model.state_dict()
    excluded = exclude_keys or []
    compatible_state = {
        key: value
        for key, value in source_state.items()
        if key in target_state
        and target_state[key].shape == value.shape
        and not any(excluded_key in key for excluded_key in excluded)
    }
    target_state.update(compatible_state)
    model.load_state_dict(target_state)
    return len(compatible_state)


def freeze_parameters(model: nn.Module, freeze_prefixes: list[str]) -> int:
    frozen = 0
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in freeze_prefixes):
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    return frozen


def forward_for_training(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if not hasattr(model, "forward_train"):
        return model(images), None, None

    outputs = model.forward_train(images)
    if not isinstance(outputs, tuple):
        return outputs, None, None
    if len(outputs) == 3:
        return outputs
    if len(outputs) == 2:
        prediction, pose_prediction = outputs
        return prediction, pose_prediction, None
    raise RuntimeError("Unexpected forward_train output shape")


def build_target_heatmaps(
    landmarks: torch.Tensor,
    height: int,
    width: int,
    sigma: float,
    normalize: bool,
) -> torch.Tensor:
    target_x = landmarks[..., 0] * (width - 1)
    target_y = landmarks[..., 1] * (height - 1)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=landmarks.device, dtype=landmarks.dtype),
        torch.arange(width, device=landmarks.device, dtype=landmarks.dtype),
        indexing="ij",
    )
    grid_x = grid_x.view(1, 1, height, width)
    grid_y = grid_y.view(1, 1, height, width)
    squared_distance = (grid_x - target_x.unsqueeze(-1).unsqueeze(-1)).pow(2) + (
        grid_y - target_y.unsqueeze(-1).unsqueeze(-1)
    ).pow(2)
    heatmaps = torch.exp(-squared_distance / (2 * sigma * sigma))
    if normalize:
        heatmaps = heatmaps / heatmaps.flatten(2).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-6)
    return heatmaps


def build_pip_targets(landmarks: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_x = (landmarks[..., 0] * width).clamp(min=0.0, max=width - 1e-4)
    target_y = (landmarks[..., 1] * height).clamp(min=0.0, max=height - 1e-4)
    cell_x = torch.floor(target_x).long().clamp(min=0, max=width - 1)
    cell_y = torch.floor(target_y).long().clamp(min=0, max=height - 1)
    class_target = cell_y * width + cell_x
    offset_x = target_x - cell_x.to(landmarks.dtype)
    offset_y = target_y - cell_y.to(landmarks.dtype)
    return class_target, offset_x, offset_y


def compute_total_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: dict,
    geometry_edges: torch.Tensor,
    pose_prediction: torch.Tensor | None = None,
    pose_target: torch.Tensor | None = None,
    heatmaps: torch.Tensor | None = None,
    teacher_prediction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_config = config["loss"]
    coordinate_loss = wing_loss(
        prediction,
        target,
        width=float(loss_config["wing_w"]),
        epsilon=float(loss_config["wing_epsilon"]),
    )
    geometric = geometry_loss(prediction, target, geometry_edges)
    coord_weight = float(loss_config.get("coord_weight", 1.0))
    total = coord_weight * coordinate_loss + float(loss_config["geometry_weight"]) * geometric
    distill_loss = torch.zeros((), device=prediction.device)
    distill_weight = float(loss_config.get("distill_weight", 0.0))
    if teacher_prediction is not None and distill_weight > 0.0:
        distill_loss = wing_loss(
            prediction,
            teacher_prediction,
            width=float(loss_config["wing_w"]),
            epsilon=float(loss_config["wing_epsilon"]),
        )
        total = total + distill_weight * distill_loss
    pose_loss = torch.zeros((), device=prediction.device)
    if pose_prediction is not None and pose_target is not None:
        pose_loss = torch.nn.functional.smooth_l1_loss(pose_prediction, pose_target)
        total = total + float(loss_config.get("pose_weight", 0.0)) * pose_loss
    heatmap_loss = torch.zeros((), device=prediction.device)
    pip_cls_loss = torch.zeros((), device=prediction.device)
    pip_offset_loss = torch.zeros((), device=prediction.device)
    if isinstance(heatmaps, dict) and {"pip_cls", "pip_x", "pip_y"}.issubset(heatmaps.keys()):
        cls_logits = heatmaps["pip_cls"]
        x_offset = heatmaps["pip_x"]
        y_offset = heatmaps["pip_y"]
        height, width = cls_logits.shape[-2:]
        class_target, target_x_offset, target_y_offset = build_pip_targets(target, height, width)

        pip_cls_loss = torch.nn.functional.cross_entropy(
            cls_logits.view(-1, height * width),
            class_target.view(-1),
        )
        flat_x = x_offset.view(x_offset.size(0), x_offset.size(1), -1)
        flat_y = y_offset.view(y_offset.size(0), y_offset.size(1), -1)
        gathered_x = torch.gather(flat_x, 2, class_target.unsqueeze(-1)).squeeze(-1)
        gathered_y = torch.gather(flat_y, 2, class_target.unsqueeze(-1)).squeeze(-1)
        pip_offset_loss = torch.nn.functional.smooth_l1_loss(gathered_x.sigmoid(), target_x_offset) + torch.nn.functional.smooth_l1_loss(gathered_y.sigmoid(), target_y_offset)
        total = total + float(loss_config.get("pip_cls_weight", 1.0)) * pip_cls_loss + float(loss_config.get("pip_offset_weight", 1.0)) * pip_offset_loss
    if heatmaps is not None:
        if isinstance(heatmaps, dict):
            heatmaps = None
        else:
            heatmap_loss_type = str(loss_config.get("heatmap_loss", "kl"))
            target_heatmaps = build_target_heatmaps(
                target,
                height=heatmaps.shape[-2],
                width=heatmaps.shape[-1],
                sigma=float(loss_config.get("heatmap_sigma", 1.5)),
                normalize=heatmap_loss_type == "kl",
            )
            if heatmap_loss_type == "mse":
                heatmap_loss = torch.nn.functional.mse_loss(torch.sigmoid(heatmaps), target_heatmaps)
            elif heatmap_loss_type == "mse_raw":
                heatmap_loss = torch.nn.functional.mse_loss(heatmaps, target_heatmaps)
            elif heatmap_loss_type == "awing":
                heatmap_loss = adaptive_wing_loss(
                    torch.sigmoid(heatmaps),
                    target_heatmaps,
                    omega=float(loss_config.get("awing_omega", 14.0)),
                    epsilon=float(loss_config.get("awing_epsilon", 0.5)),
                    alpha=float(loss_config.get("awing_alpha", 2.1)),
                    theta=float(loss_config.get("awing_theta", 0.5)),
                )
            else:
                predicted_log_probs = torch.log_softmax(heatmaps.flatten(2), dim=-1)
                target_probs = target_heatmaps.flatten(2)
                heatmap_loss = torch.nn.functional.kl_div(predicted_log_probs, target_probs, reduction="batchmean")
            total = total + float(loss_config.get("heatmap_weight", 0.0)) * heatmap_loss
    pieces = {
        "coord": float(coordinate_loss.detach().item()),
        "geom": float(geometric.detach().item()),
        "pose": float(pose_loss.detach().item()),
        "heatmap": float(heatmap_loss.detach().item()),
        "pip_cls": float(pip_cls_loss.detach().item()),
        "pip_offset": float(pip_offset_loss.detach().item()),
        "distill": float(distill_loss.detach().item()),
    }
    return total, pieces


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict,
    geometry_edges: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["landmarks"].to(device, non_blocking=True)
            pose_target = batch.get("pose")
            pose_target = pose_target.to(device, non_blocking=True) if pose_target is not None else None
            prediction, pose_prediction, heatmaps = forward_for_training(model, images)
            loss, _ = compute_total_loss(prediction, target, config, geometry_edges, pose_prediction, pose_target, heatmaps)
            losses.append(float(loss.item()))
            predictions.append(prediction.cpu())
            targets.append(target.cpu())
    merged_predictions = torch.cat(predictions, dim=0)
    merged_targets = torch.cat(targets, dim=0)
    metrics = compute_metrics(
        merged_predictions,
        merged_targets,
        eye_groups=config["data"]["metric_eye_groups"],
    )
    return sum(losses) / max(1, len(losses)), metrics


def save_checkpoint(path: Path, model: nn.Module, config: dict, epoch: int, metrics: dict[str, float]) -> None:
    payload = {
        "model": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a scratch face landmark model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--note", default="baseline")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    set_seed(int(config["seed"]))
    torch.backends.cudnn.benchmark = True

    output_root = ensure_dir(config["system"]["output_root"])
    run_dir = ensure_dir(output_root / config["run_name"])
    history_path = run_dir / "history.csv"
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    preview_path = run_dir / "preview_best.png"
    plot_path = run_dir / "history.png"
    log_path = Path("optimization_log.md")

    dataloaders = create_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    ssl_checkpoint = config["train"].get("ssl_checkpoint")
    if ssl_checkpoint:
        ssl_payload = torch.load(ssl_checkpoint, map_location="cpu", weights_only=False)
        ssl_state = ssl_payload.get("backbone", ssl_payload.get("model", ssl_payload))
        exclude_keys = config["train"].get("ssl_exclude_keys", [])
        loaded_count = load_compatible_state_dict(model, ssl_state, exclude_keys=exclude_keys)
        print(f"loaded_ssl_tensors={loaded_count} checkpoint={ssl_checkpoint}")
    if bool(config["train"].get("translation_adapter", False)):
        adapter_hidden_dim = int(config["train"].get("translation_hidden_dim", 256))
        adapter_scale = float(config["train"].get("translation_scale", 0.15))
        model = LandmarkTranslationWrapper(
            model,
            num_landmarks=int(config["data"]["num_landmarks"]),
            hidden_dim=adapter_hidden_dim,
            residual_scale=adapter_scale,
        ).to(device)
        print(f"translation_adapter=true hidden_dim={adapter_hidden_dim} residual_scale={adapter_scale}")
    freeze_prefixes = config["train"].get("freeze_prefixes", [])
    if freeze_prefixes:
        frozen_count = freeze_parameters(model, freeze_prefixes)
        print(f"frozen_parameters={frozen_count} prefixes={freeze_prefixes}")
    finetune_checkpoint = config["train"].get("finetune_checkpoint")
    if finetune_checkpoint:
        ft_payload = torch.load(finetune_checkpoint, map_location="cpu", weights_only=False)
        loaded_count = load_compatible_state_dict(model, ft_payload.get("model", ft_payload))
        print(f"loaded_finetune_checkpoint={loaded_count} tensors from {finetune_checkpoint}")
    elif bool(config["train"].get("init_mean_shape", False)) and hasattr(model, "initialize_landmark_bias"):
        mean_shape = estimate_mean_shape(dataloaders.valid.dataset)
        model.initialize_landmark_bias(mean_shape)
    ema = ModelEma(model, decay=float(config["train"]["ema_decay"]))
    geometry_edges = build_geometry_edges(int(config["data"]["num_landmarks"]))

    teacher_model: nn.Module | None = None
    teacher_checkpoint = config["train"].get("distill_teacher_checkpoint")
    if teacher_checkpoint:
        teacher_payload = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
        teacher_config = teacher_payload["config"]
        teacher_model = build_model(teacher_config).to(device)
        teacher_model.load_state_dict(teacher_payload["model"])
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        print(f"loaded distillation teacher: {teacher_checkpoint}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    steps_per_epoch = max(1, len(dataloaders.train))
    scheduler_name = str(config["train"].get("scheduler", "onecycle"))
    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, steps_per_epoch * int(config["train"]["epochs"])),
            eta_min=float(config["train"].get("min_lr", 1e-6)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(config["train"]["lr"]),
            epochs=int(config["train"]["epochs"]),
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15,
            div_factor=10.0,
            final_div_factor=100.0,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda")
    use_ema = bool(config["train"].get("use_ema", True))

    history: list[dict[str, float]] = []
    select_metric = str(config["train"].get("select_metric", "valid_acc_008"))
    select_mode = str(config["train"].get("select_mode", "max"))
    best_valid = float("inf") if select_mode == "min" else float("-inf")
    best_epoch = 0
    best_test_metrics: dict[str, float] = {}
    best_valid_metrics: dict[str, float] = {}
    csv_header_written = history_path.exists()
    log_interval_steps = int(config["train"].get("log_interval_steps", 0))

    # Early stopping
    early_stop_patience = int(config["train"].get("early_stop_patience", 0))
    epochs_without_improvement = 0

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        epoch_start = perf_counter()
        model.train()
        running_loss = 0.0
        running_coord = 0.0
        running_geom = 0.0
        running_pose = 0.0
        running_heatmap = 0.0
        running_pip_cls = 0.0
        running_pip_offset = 0.0
        running_distill = 0.0
        progress = tqdm(dataloaders.train, desc=f"epoch {epoch:02d}", leave=False)
        for step_index, batch in enumerate(progress, start=1):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["landmarks"].to(device, non_blocking=True)
            pose_target = batch.get("pose")
            pose_target = pose_target.to(device, non_blocking=True) if pose_target is not None else None
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=bool(config["train"]["amp"]) and device.type == "cuda"):
                prediction, pose_prediction, heatmaps = forward_for_training(model, images)
                teacher_prediction: torch.Tensor | None = None
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_prediction, _, _ = forward_for_training(teacher_model, images)
                        teacher_prediction = teacher_prediction.detach()
                loss, parts = compute_total_loss(
                    prediction,
                    target,
                    config,
                    geometry_edges,
                    pose_prediction,
                    pose_target,
                    heatmaps,
                    teacher_prediction=teacher_prediction,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            if use_ema:
                ema.update(model)

            running_loss += float(loss.item())
            running_coord += parts["coord"]
            running_geom += parts["geom"]
            running_pose += parts["pose"]
            running_heatmap += parts["heatmap"]
            running_pip_cls += parts["pip_cls"]
            running_pip_offset += parts["pip_offset"]
            running_distill += parts.get("distill", 0.0)
            progress.set_postfix(
                loss=f"{running_loss / max(1, step_index):.4f}",
                pose=f"{running_pose / max(1, step_index):.4f}",
                lr=f"{(scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']):.5f}",
            )
            if log_interval_steps and (step_index % log_interval_steps == 0 or step_index == len(dataloaders.train)):
                print(
                    f"epoch={epoch} step={step_index}/{len(dataloaders.train)} "
                    f"loss={running_loss / step_index:.5f} coord={running_coord / step_index:.5f} "
                    f"geom={running_geom / step_index:.5f} pose={running_pose / step_index:.5f} "
                    f"heatmap={running_heatmap / step_index:.5f} pip_cls={running_pip_cls / step_index:.5f} "
                    f"pip_offset={running_pip_offset / step_index:.5f} "
                    f"distill={running_distill / step_index:.5f} "
                    f"lr={(scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']):.6f}"
                )

        train_loss = running_loss / max(1, len(dataloaders.train))
        eval_model = ema.module if use_ema else model
        valid_loss, valid_metrics = evaluate(eval_model, dataloaders.valid, device, config, geometry_edges)
        test_loss, test_metrics = evaluate(eval_model, dataloaders.test, device, config, geometry_edges)
        epoch_seconds = perf_counter() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "test_loss": test_loss,
            "train_coord": running_coord / max(1, len(dataloaders.train)),
            "train_geom": running_geom / max(1, len(dataloaders.train)),
            "train_pose": running_pose / max(1, len(dataloaders.train)),
            "train_heatmap": running_heatmap / max(1, len(dataloaders.train)),
            "train_pip_cls": running_pip_cls / max(1, len(dataloaders.train)),
            "train_pip_offset": running_pip_offset / max(1, len(dataloaders.train)),
            "epoch_seconds": epoch_seconds,
            **flatten_metrics("valid", valid_metrics),
            **flatten_metrics("test", test_metrics),
        }
        history.append(epoch_record)

        with history_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(epoch_record.keys()))
            if not csv_header_written:
                writer.writeheader()
                csv_header_written = True
            writer.writerow(epoch_record)

        current_selection_value = epoch_record[select_metric]
        is_better = current_selection_value < best_valid if select_mode == "min" else current_selection_value > best_valid
        if is_better:
            best_valid = current_selection_value
            best_epoch = epoch
            epochs_without_improvement = 0
            best_test_metrics = test_metrics
            best_valid_metrics = valid_metrics
            save_checkpoint(
                best_path,
                eval_model,
                config,
                epoch,
                {**valid_metrics, **flatten_metrics("test", test_metrics), "selection_metric": select_metric, "selection_value": current_selection_value},
            )
            render_predictions(eval_model, dataloaders.valid, device, preview_path, int(config["eval"]["viz_samples"]))
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch} train_loss={train_loss:.5f} valid_acc_008={valid_metrics['acc_008']:.3f} "
            f"test_acc_008={test_metrics['acc_008']:.3f} nme={valid_metrics['nme']:.5f} time={epoch_seconds:.1f}s"
            + (f" no_improve={epochs_without_improvement}" if early_stop_patience > 0 else "")
        )

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}: no improvement for {early_stop_patience} epochs (best epoch={best_epoch})")
            break

    save_history_plot(history, plot_path)
    summary = {
        "run_name": config["run_name"],
        "note": args.note,
        "best_epoch": best_epoch,
        "selection_metric": select_metric,
        "selection_mode": select_mode,
        "selection_value": best_valid,
        "best_valid_acc_008": best_valid_metrics.get("acc_008", 0.0),
        "best_test_acc_008": best_test_metrics.get("acc_008", 0.0),
        "best_test_acc_005": best_test_metrics.get("acc_005", 0.0),
        "best_test_nme": best_test_metrics.get("nme", 0.0),
        "parameter_count": count_parameters(model),
        "num_landmarks": int(config["data"]["num_landmarks"]),
        "estimated_int8_size_mb": parameter_size_mb(model, bytes_per_weight=1),
        "estimated_fp32_size_mb": parameter_size_mb(model, bytes_per_weight=4),
        "preview_path": str(preview_path),
        "history_path": str(history_path),
    }
    write_json(summary_path, summary)
    append_optimization_log(log_path, stage="train", run_name=config["run_name"], note=args.note, summary=summary)
    print(summary)


if __name__ == "__main__":
    main()
