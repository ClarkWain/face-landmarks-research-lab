from __future__ import annotations

import argparse
import copy
import io
import json
import zipfile
from pathlib import Path
from time import perf_counter

import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from landmarklab.core import append_optimization_log, ensure_dir, load_config, set_seed, write_json
from landmarklab.model import build_model


class CelebAZipViewsDataset(Dataset):
    def __init__(self, zip_path: str | Path | None, image_root: str | Path | None, image_size: int, max_samples: int | None) -> None:
        self.zip_path = Path(zip_path) if zip_path else None
        self.image_root = Path(image_root) if image_root else None
        if self.zip_path is not None:
            with zipfile.ZipFile(self.zip_path) as archive:
                self.members = [name for name in archive.namelist() if name.lower().endswith(".jpg")]
        elif self.image_root is not None:
            self.members = [str(path) for path in sorted(self.image_root.rglob("*.jpg"))]
        else:
            raise ValueError("Either zip_path or image_root must be provided")
        if max_samples is not None:
            self.members = self.members[:max_samples]
        color_jitter = T.ColorJitter(0.4, 0.4, 0.2, 0.1)
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.archive: zipfile.ZipFile | None = None


    def __len__(self) -> int:
        return len(self.members)


    def _get_archive(self) -> zipfile.ZipFile:
        if self.zip_path is None:
            raise RuntimeError("Archive access requested for a directory-backed dataset")
        if self.archive is None:
            self.archive = zipfile.ZipFile(self.zip_path)
        return self.archive


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.zip_path is not None:
            archive = self._get_archive()
            with archive.open(self.members[index]) as file:
                image = Image.open(io.BytesIO(file.read())).convert("RGB")
        else:
            image = Image.open(self.members[index]).convert("RGB")
        return self.transform(image), self.transform(image)


class SSLWrapper(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.backbone = build_model(config)
        image_size = int(config["data"]["image_size"])
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feature = self.backbone.extract_features(dummy)[-1]
            feature_dim = self.backbone.global_pool(feature).flatten(1).shape[1]

        projector_dim = int(config["ssl"]["projector_dim"])
        predictor_dim = int(config["ssl"]["predictor_dim"])
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim, affine=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(projector_dim, predictor_dim),
            nn.BatchNorm1d(predictor_dim),
            nn.ReLU(inplace=True),
            nn.Linear(predictor_dim, projector_dim),
        )


    def encode(self, images: torch.Tensor) -> torch.Tensor:
        feature = self.backbone.extract_features(images)[-1]
        return self.backbone.global_pool(feature).flatten(1)


    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_a = self.projector(self.encode(view_a))
        z_b = self.projector(self.encode(view_b))
        p_a = self.predictor(z_a)
        p_b = self.predictor(z_b)
        return p_a, p_b, z_a.detach(), z_b.detach()


def negative_cosine_similarity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = torch.nn.functional.normalize(prediction, dim=1)
    target = torch.nn.functional.normalize(target, dim=1)
    return -(prediction * target).sum(dim=1).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-supervised pretraining on CelebA zip")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--note", default="ssl_baseline")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    set_seed(int(config["seed"]))
    torch.backends.cudnn.benchmark = True

    run_dir = ensure_dir(Path(config["system"]["output_root"]) / config["run_name"])
    dataset = CelebAZipViewsDataset(
        config["data"].get("zip_path"),
        config["data"].get("image_root"),
        image_size=int(config["data"]["image_size"]),
        max_samples=config["data"].get("max_samples"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["ssl"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLWrapper(config).to(device)
    ema_backbone = copy.deepcopy(model.backbone).eval()
    for parameter in ema_backbone.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["ssl"]["lr"]),
        weight_decay=float(config["ssl"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, len(loader) * int(config["ssl"]["epochs"])))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["ssl"]["amp"]) and device.type == "cuda")

    history: list[dict[str, float]] = []
    log_interval_steps = int(config["ssl"].get("log_interval_steps", 100))
    for epoch in range(1, int(config["ssl"]["epochs"]) + 1):
        epoch_start = perf_counter()
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"ssl {epoch:02d}", leave=False)
        for step_index, (view_a, view_b) in enumerate(progress, start=1):
            view_a = view_a.to(device, non_blocking=True)
            view_b = view_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=bool(config["ssl"]["amp"]) and device.type == "cuda"):
                p_a, p_b, z_a, z_b = model(view_a, view_b)
                loss = 0.5 * (negative_cosine_similarity(p_a, z_b) + negative_cosine_similarity(p_b, z_a))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{running_loss / step_index:.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")
            if log_interval_steps and (step_index % log_interval_steps == 0 or step_index == len(loader)):
                print(f"ssl_epoch={epoch} step={step_index}/{len(loader)} loss={running_loss / step_index:.5f} lr={scheduler.get_last_lr()[0]:.6f}")

        epoch_loss = running_loss / max(1, len(loader))
        history.append({"epoch": epoch, "loss": epoch_loss, "epoch_seconds": perf_counter() - epoch_start})
        ema_backbone.load_state_dict(model.backbone.state_dict())
        print(f"ssl_epoch={epoch} loss={epoch_loss:.5f} time={history[-1]['epoch_seconds']:.1f}s")
        torch.save(
            {"backbone": model.backbone.state_dict(), "config": config, "history": history},
            run_dir / "ssl_backbone_last.pt",
        )

    checkpoint_path = run_dir / "ssl_backbone.pt"
    torch.save({"backbone": model.backbone.state_dict(), "config": config, "history": history}, checkpoint_path)
    summary = {
        "run_name": config["run_name"],
        "note": args.note,
        "epochs": int(config["ssl"]["epochs"]),
        "last_loss": history[-1]["loss"] if history else None,
        "checkpoint": str(checkpoint_path),
    }
    write_json(run_dir / "ssl_summary.json", summary)
    append_optimization_log(Path("optimization_log.md"), stage="ssl", run_name=config["run_name"], note=args.note, summary=summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()