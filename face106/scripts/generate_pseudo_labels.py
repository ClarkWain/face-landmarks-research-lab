"""Generate 106-point pseudo labels on WFLW images using a trained HRNet teacher model.

This script:
1. Loads a trained 106-point HRNet model (teacher)
2. Processes WFLW images through the teacher to get 106-point predictions
3. Saves pseudo labels in a format compatible with the training pipeline

Usage:
    cd face106
    python scripts/generate_pseudo_labels.py --config configs/lapa_hrnet_w18_heatmap_ft.yaml --run runs/lapa_hrnet_w18_heatmap_ft
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarklab.core import load_config
from landmarklab.model import build_model


class WFLWImageDataset(Dataset):
    """Simple dataset to load WFLW images for pseudo-label generation."""
    
    def __init__(self, image_paths: list[Path], image_size: int = 256):
        self.image_paths = image_paths
        self.image_size = image_size
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        
        # Resize to expected input size
        image = TF.resize(image, [self.image_size, self.image_size], 
                         interpolation=InterpolationMode.BILINEAR, antialias=True)
        
        # Convert to tensor and normalize
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        return {
            "image": tensor,
            "path": str(path),
            "original_size": torch.tensor([image.size[0], image.size[1]]),
        }


def load_wflw_image_paths(wflw_root: Path, split: str = "train") -> list[Path]:
    """Load image paths from WFLW dataset."""
    if split == "train":
        img_dir = wflw_root / "train_data" / "imgs"
    else:
        img_dir = wflw_root / "test_data" / "imgs"
    
    if not img_dir.exists():
        raise FileNotFoundError(f"WFLW image directory not found: {img_dir}")
    
    paths = sorted(img_dir.glob("*.png"))
    if not paths:
        # Try .jpg if no .png found
        paths = sorted(img_dir.glob("*.jpg"))
    
    return paths


@torch.no_grad()
def generate_pseudo_labels(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Generate pseudo labels for all images in the dataloader.
    
    Returns:
        Dictionary mapping image path to predicted landmarks array (106, 2)
    """
    model.eval()
    results = {}
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        paths = batch["path"]
        
        # Forward pass
        predictions = model(images)  # (B, 106, 2) in normalized coords
        
        # Store results
        for i, (pred, path) in enumerate(zip(predictions, paths)):
            results[path] = pred.cpu().numpy()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Processed {len(results)} images...")
    
    return results


def save_pseudo_labels(labels: dict[str, np.ndarray], output_path: Path):
    """Save pseudo labels to a CSV file compatible with our training pipeline.
    
    Format: image_path, 212 values (106 points * 2 coords), source
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for image_path, landmarks in sorted(labels.items()):
            # Flatten landmarks to [x1, y1, x2, y2, ...]
            flat_coords = landmarks.flatten().tolist()
            writer.writerow([image_path] + flat_coords + ["pseudo_hrnet"])
    
    print(f"Saved {len(labels)} pseudo labels to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 106-point pseudo labels on WFLW images")
    parser.add_argument("--config", required=True, help="Path to trained model config YAML")
    parser.add_argument("--run", required=True, help="Path to trained model run directory")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    config = load_config(args.config)
    run_dir = Path(args.run)
    checkpoint_path = run_dir / "best.pt"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    
    # Load WFLW image paths
    wflw_root = Path(config["data"].get("root", "../data")) / "wflw_extracted"
    
    output_dir = Path("data/wflw_pseudo_106")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "test"]:
        print(f"\nProcessing {split} split...")
        
        try:
            image_paths = load_wflw_image_paths(wflw_root, split)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        
        if not image_paths:
            print(f"No images found for {split} split")
            continue
        
        print(f"Found {len(image_paths)} images")
        
        # Create dataloader
        dataset = WFLWImageDataset(image_paths, image_size=args.image_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=4, pin_memory=True)
        
        # Generate pseudo labels
        print("Generating pseudo labels...")
        pseudo_labels = generate_pseudo_labels(model, dataloader, device)
        
        # Save results
        output_path = output_dir / f"wflw_{split}_pseudo_106.csv"
        save_pseudo_labels(pseudo_labels, output_path)
    
    print("\nPseudo-label generation complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
