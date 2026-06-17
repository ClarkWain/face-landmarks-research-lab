"""Compute PCA shape basis from LaPa training set landmarks.

Outputs face106/data/lapa_pca_shape.npz with:
  mean_shape: (106, 2) float32
  basis: (K, 106, 2) float32 (top-K eigenvectors)
  eigenvalues: (K,) float32 (explained variance per basis)
  cumulative_var: (K,) float32 (cumulative explained variance ratio)

Each landmark is normalized to a unit bbox (0..1) before PCA so the basis is
scale/translation invariant.

Usage:
    cd face106
    py -3.12 scripts/compute_pca_shape.py --num-basis 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_lapa_landmarks(lapa_root: Path, split: str = "train") -> np.ndarray:
    """Load all 106-pt landmarks from a LaPa split. Returns (N, 106, 2) pixel coords."""
    landmarks_dir = lapa_root / split / "landmarks"
    files = sorted(landmarks_dir.glob("*.txt"))
    print(f"[load] {split}: {len(files)} files")
    samples = []
    for f in files:
        arr = np.loadtxt(f, skiprows=1)  # first line is "106"
        if arr.shape != (106, 2):
            print(f"  skip {f.name}: shape={arr.shape}")
            continue
        samples.append(arr.astype(np.float32))
    return np.stack(samples)


def normalize_to_unit_bbox(landmarks: np.ndarray) -> np.ndarray:
    """Normalize each sample's landmarks to unit bbox [0, 1].

    landmarks: (N, 106, 2)
    returns: (N, 106, 2) with x ∈ [0, 1], y ∈ [0, 1]
    """
    out = np.empty_like(landmarks)
    for i, s in enumerate(landmarks):
        x_min, y_min = s.min(axis=0)
        x_max, y_max = s.max(axis=0)
        # use the larger dimension to keep aspect ratio
        side = max(x_max - x_min, y_max - y_min)
        if side <= 0:
            out[i] = 0.5
            continue
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        # center then scale so bbox half-side = 0.5
        out[i, :, 0] = (s[:, 0] - cx) / side + 0.5
        out[i, :, 1] = (s[:, 1] - cy) / side + 0.5
    return out


def compute_pca(samples_normalized: np.ndarray, num_basis: int):
    """Compute PCA on flattened landmark vectors."""
    n_samples = samples_normalized.shape[0]
    flat = samples_normalized.reshape(n_samples, -1)  # (N, 212)
    mean = flat.mean(axis=0)  # (212,)
    centered = flat - mean[None, :]  # (N, 212)

    # SVD on centered matrix: centered = U * diag(s) * V^T
    # eigenvectors of covariance = rows of V (columns of V^T)
    print(f"[pca] computing SVD on ({n_samples}, 212)...")
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (s ** 2) / (n_samples - 1)  # population covariance eigenvalues
    total_var = eigenvalues.sum()
    explained_ratio = eigenvalues / total_var
    cumulative = np.cumsum(explained_ratio)

    basis = vt[:num_basis]  # (K, 212)
    eigvals = eigenvalues[:num_basis]
    cum = cumulative[:num_basis]

    print(f"[pca] cumulative variance for K=1..{num_basis}:")
    for i, c in enumerate(cum):
        print(f"  K={i + 1:2d}: {c * 100:.2f}%")

    # Reshape basis to (K, 106, 2) and mean to (106, 2)
    mean_shape = mean.reshape(106, 2).astype(np.float32)
    basis = basis.reshape(num_basis, 106, 2).astype(np.float32)
    return mean_shape, basis, eigvals.astype(np.float32), cum.astype(np.float32)


def reconstruct_error(samples_normalized: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> tuple[float, float]:
    """Reconstruct each sample using mean + best-fit linear combination of basis vectors."""
    flat = samples_normalized.reshape(samples_normalized.shape[0], -1)  # (N, 212)
    flat_mean = mean.reshape(-1)  # (212,)
    flat_basis = basis.reshape(basis.shape[0], -1)  # (K, 212)

    centered = flat - flat_mean
    # least-squares projection onto basis
    coefs = centered @ flat_basis.T  # (N, K)
    reconstructed = coefs @ flat_basis + flat_mean  # (N, 212)
    err = (reconstructed - flat).reshape(-1, 106, 2)
    # NME-like: per-sample L2 distance averaged over points
    per_sample_nme = np.linalg.norm(err, axis=-1).mean(axis=-1)  # (N,)
    return float(per_sample_nme.mean()), float(per_sample_nme.std())


def main():
    parser = argparse.ArgumentParser(description="Compute PCA shape basis from LaPa")
    parser.add_argument("--lapa-root", default="../data/LaPa", help="Root containing train/landmarks/*.txt")
    parser.add_argument("--num-basis", type=int, default=16, help="Number of PCA basis vectors to retain")
    parser.add_argument("--output", default="data/lapa_pca_shape.npz")
    args = parser.parse_args()

    lapa_root = Path(args.lapa_root).resolve()
    print(f"[main] LaPa root: {lapa_root}")

    landmarks = load_lapa_landmarks(lapa_root, "train")
    print(f"[main] loaded {len(landmarks)} train samples, shape={landmarks.shape}")

    normalized = normalize_to_unit_bbox(landmarks)
    print(f"[main] normalized: range=[{normalized.min():.3f}, {normalized.max():.3f}]")

    mean_shape, basis, eigvals, cum = compute_pca(normalized, args.num_basis)
    print(f"[main] mean_shape: {mean_shape.shape} range=[{mean_shape.min():.3f}, {mean_shape.max():.3f}]")
    print(f"[main] basis: {basis.shape}")

    err_mean, err_std = reconstruct_error(normalized, mean_shape, basis)
    print(f"[main] reconstruction NME (relative to unit-bbox): {err_mean * 100:.4f}% ± {err_std * 100:.4f}%")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        mean_shape=mean_shape,
        basis=basis,
        eigenvalues=eigvals,
        cumulative_var=cum,
    )
    print(f"[main] saved: {output} ({output.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
