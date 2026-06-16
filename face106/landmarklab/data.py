from __future__ import annotations

import csv
import os
import random
import tarfile
from dataclasses import dataclass
import io
from pathlib import Path
import urllib.request
import zipfile

import torch
from PIL import Image, ImageFilter
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


FLIP_ORDER_68 = torch.tensor(
    [
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
        27, 28, 29, 30, 35, 34, 33, 32, 31,
        45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40,
        54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,
        64, 63, 62, 61, 60, 67, 66, 65,
    ],
    dtype=torch.long,
)

WFLW_FLIP_PAIRS = [
    (0, 32), (1, 31), (2, 30), (3, 29), (4, 28), (5, 27), (6, 26), (7, 25),
    (8, 24), (9, 23), (10, 22), (11, 21), (12, 20), (13, 19), (14, 18), (15, 17),
    (33, 46), (34, 45), (35, 44), (36, 43), (37, 42), (38, 50), (39, 49),
    (40, 48), (41, 47), (55, 59), (56, 58), (60, 72), (61, 71), (62, 70),
    (63, 69), (64, 68), (65, 75), (66, 74), (67, 73), (76, 82), (77, 81),
    (78, 80), (83, 87), (84, 86), (88, 92), (89, 91), (93, 95), (96, 97),
]


def _build_flip_order(num_points: int, flip_pairs: list[tuple[int, int]]) -> torch.Tensor:
    order = list(range(num_points))
    for left, right in flip_pairs:
        order[left] = right
        order[right] = left
    return torch.tensor(order, dtype=torch.long)


FLIP_ORDER_98 = _build_flip_order(98, WFLW_FLIP_PAIRS)

TRAIN_FOLDERS = [
    ("afw", None),
    ("helen", "trainset"),
    ("lfpw", "trainset"),
]

TEST_FOLDERS = [
    ("ibug", None),
    ("helen", "testset"),
    ("lfpw", "testset"),
]


@dataclass
class DataBundle:
    train: DataLoader
    valid: DataLoader
    test: DataLoader


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def _find_dataset_root(base_dir: Path) -> Path:
    candidates = [base_dir] + [path for path in base_dir.iterdir() if path.is_dir()]
    for candidate in candidates:
        names = {path.name.lower() for path in candidate.iterdir() if path.is_dir()}
        if {"afw", "helen", "ibug", "lfpw"}.issubset(names):
            return candidate
        if {"01_indoor", "02_outdoor"}.issubset(names):
            return candidate
        if "300w" in names:
            nested = candidate / "300W"
            if nested.exists() and nested.is_dir():
                return nested
    raise FileNotFoundError(f"Unable to locate the extracted 300W root under {base_dir}")


def _prepare_300w(root: str | Path, *, download: bool, archive_url: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "300w_dataset.zip"
    extract_dir = root / "300w_extracted"

    if archive_path.exists() and not zipfile.is_zipfile(archive_path):
        archive_path.unlink()

    if not archive_path.exists():
        if not download:
            raise FileNotFoundError(f"Missing a valid {archive_path}; set data.download=true to fetch the dataset")
        _download_file(archive_url, archive_path)

    if not zipfile.is_zipfile(archive_path):
        if archive_path.exists():
            archive_path.unlink()
        raise RuntimeError(f"Downloaded archive is invalid: {archive_path}")

    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _find_dataset_root(extract_dir)
    except FileNotFoundError:
        print(f"extracting {archive_path} -> {extract_dir}")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
        return _find_dataset_root(extract_dir)


def _prepare_wflw_augmented(root: str | Path, *, download: bool, train_url: str, test_url: str) -> tuple[Path, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    extract_dir = root / "wflw_extracted"
    train_dir = extract_dir / "train_data"
    test_dir = extract_dir / "test_data"
    if (train_dir / "labels.csv").exists() and (test_dir / "labels.csv").exists():
        return train_dir, test_dir

    train_archive = root / "train_data.tar.gz"
    test_archive = root / "test_data.tar.gz"

    for archive_path, url in ((train_archive, train_url), (test_archive, test_url)):
        if archive_path.exists() and not tarfile.is_tarfile(archive_path):
            archive_path.unlink()
        if not archive_path.exists():
            if not download:
                raise FileNotFoundError(f"Missing a valid {archive_path}; set data.download=true to fetch the dataset")
            _download_file(url, archive_path)
        if not tarfile.is_tarfile(archive_path):
            if archive_path.exists():
                archive_path.unlink()
            raise RuntimeError(f"Downloaded archive is invalid: {archive_path}")

    if not train_dir.exists() or not (train_dir / "labels.csv").exists():
        train_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"extracting {train_archive} -> {extract_dir}")
        with tarfile.open(train_archive, "r:gz") as archive:
            archive.extractall(extract_dir)
    if not test_dir.exists() or not (test_dir / "labels.csv").exists():
        test_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"extracting {test_archive} -> {extract_dir}")
        with tarfile.open(test_archive, "r:gz") as archive:
            archive.extractall(extract_dir)

    return train_dir, test_dir


def _load_wflw_rows(split_dir: Path) -> list[tuple[Path, Tensor, Tensor]]:
    samples: list[tuple[Path, Tensor, Tensor]] = []
    with (split_dir / "labels.csv").open("r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) != 206:
                raise RuntimeError(f"Unexpected WFLW row width in {split_dir / 'labels.csv'}: {len(row)}")
            coords = torch.tensor([float(value) for value in row[:196]], dtype=torch.float32).view(98, 2)
            pose = torch.tensor([float(value) for value in row[-4:-1]], dtype=torch.float32) / 90.0
            image_path = split_dir / "imgs" / Path(row[-1]).name
            if image_path.exists():
                samples.append((image_path, coords, pose))
    if not samples:
        raise RuntimeError(f"No WFLW samples were found under {split_dir}")
    return samples


def _load_facesynthetics_zip(zip_path: str | Path, *, max_samples: int | None) -> list[tuple[str, Tensor]]:
    zip_path = Path(zip_path)
    samples: list[tuple[str, Tensor]] = []
    with zipfile.ZipFile(zip_path) as archive:
        image_names = [name for name in archive.namelist() if name.lower().endswith('.png') and not name.lower().endswith('_seg.png')]
        image_names.sort()
        if max_samples is not None:
            image_names = image_names[:max_samples]
        for image_name in image_names:
            stem = image_name[:-4]
            landmark_name = f"{stem}_ldmks.txt"
            try:
                with archive.open(landmark_name) as file:
                    rows = []
                    for line in io.TextIOWrapper(file, encoding='utf-8'):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.replace(',', ' ').split()
                        if len(parts) < 2:
                            continue
                        rows.append([float(parts[0]), float(parts[1])])
            except KeyError:
                continue
            landmarks = torch.tensor(rows, dtype=torch.float32)
            if landmarks.shape[0] >= 68:
                samples.append((image_name, landmarks[:68]))
    if not samples:
        raise RuntimeError(f"No FaceSynthetics samples were found in {zip_path}")
    return samples


def _load_pts(path: Path) -> Tensor:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(("version", "n_points", "{", "}")):
                continue
            x_coord, y_coord = line.split()
            rows.append([float(x_coord), float(y_coord)])
    landmarks = torch.tensor(rows, dtype=torch.float32)
    if landmarks.shape != (68, 2):
        raise RuntimeError(f"Expected 68 landmarks in {path}, found {tuple(landmarks.shape)}")
    return landmarks


def _load_lapa_landmarks(path: Path) -> Tensor:
    """LaPa landmark .txt: first line is the count (106), remaining 106 lines are 'x y' in pixel coords."""
    with path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    if not lines:
        raise RuntimeError(f"empty LaPa landmark file: {path}")
    expected = int(lines[0])
    coords = []
    for line in lines[1 : expected + 1]:
        parts = line.split()
        coords.append([float(parts[0]), float(parts[1])])
    landmarks = torch.tensor(coords, dtype=torch.float32)
    if landmarks.shape[0] != expected:
        raise RuntimeError(f"Expected {expected} LaPa landmarks in {path}, found {landmarks.shape[0]}")
    return landmarks


def _collect_lapa_samples(split_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = split_dir / "images"
    landmarks_dir = split_dir / "landmarks"
    if not images_dir.exists() or not landmarks_dir.exists():
        raise RuntimeError(f"LaPa split missing images/ or landmarks/ under {split_dir}")
    samples: list[tuple[Path, Path]] = []
    for landmark_path in sorted(landmarks_dir.glob("*.txt")):
        image_path = images_dir / f"{landmark_path.stem}.jpg"
        if image_path.exists():
            samples.append((image_path, landmark_path))
    if not samples:
        raise RuntimeError(f"No LaPa samples were found under {split_dir}")
    return samples


def _collect_jd_landmark_samples(root: Path) -> list[tuple[Path, Path]]:
    """Collect JD-landmark (ICME 2021 FLL3) samples. Same 106-point format as LaPa."""
    images_dir = root / "train" / "picture_mask"
    landmarks_dir = root / "train" / "landmark"
    if not images_dir.exists() or not landmarks_dir.exists():
        raise RuntimeError(f"JD-landmark missing picture_mask/ or landmark/ under {root / 'train'}")
    samples: list[tuple[Path, Path]] = []
    for landmark_path in sorted(landmarks_dir.glob("*.txt")):
        image_path = images_dir / f"{landmark_path.stem}.jpg"
        if image_path.exists():
            samples.append((image_path, landmark_path))
    if not samples:
        raise RuntimeError(f"No JD-landmark samples were found under {root}")
    return samples


def _load_pseudo_wflw_rows(csv_path: Path) -> list[tuple[Path, Tensor]]:
    """Load pseudo-labeled WFLW data from CSV: image_path, 212 coords (106*2 pixel x,y)."""
    samples: list[tuple[Path, Tensor]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 213:  # path + 212 coords (106*2)
                continue
            image_path = Path(row[0])
            if not image_path.exists():
                continue
            coords = torch.tensor([float(v) for v in row[1:213]], dtype=torch.float32).view(106, 2)
            samples.append((image_path, coords))
    return samples


class PseudoWFLWDataset(Dataset):
    """WFLW images with 106-point pseudo labels from HRNet teacher.

    samples: list of (image_path, landmarks_tensor_106x2_in_pixel_coords).
    Same processing pipeline as LaPaDataset: crop, resize, normalize to [0,1].
    """

    def __init__(
        self,
        samples: list[tuple[Path, Tensor]],
        *,
        image_size: int,
        augment: bool,
        max_samples: int | None,
        landmark_subset: str | None = None,
        crop_scale: float = 1.30,
        aug_scale_range: tuple[float, float] = (0.92, 1.18),
        aug_shift: float = 0.05,
        use_full_image: bool = False,
        enable_hflip: bool = False,
    ) -> None:
        self.samples = samples[:max_samples] if max_samples is not None else samples
        self.image_size = image_size
        self.augment = augment
        self.landmark_subset = landmark_subset
        self.crop_scale = crop_scale
        self.aug_scale_range = aug_scale_range
        self.aug_shift = aug_shift
        self.use_full_image = use_full_image
        self.enable_hflip = enable_hflip

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_crop_box(self, landmarks: Tensor, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        min_xy = landmarks.min(dim=0).values
        max_xy = landmarks.max(dim=0).values
        width, height = (max_xy - min_xy).tolist()
        center_x, center_y = ((min_xy + max_xy) * 0.5).tolist()
        scale = self.crop_scale
        shift_scale = 0.0
        if self.augment:
            scale *= random.uniform(*self.aug_scale_range)
            shift_scale = self.aug_shift
        crop_size = max(width, height) * scale
        center_x += random.uniform(-shift_scale, shift_scale) * crop_size
        center_y += random.uniform(-shift_scale, shift_scale) * crop_size
        left = max(0.0, center_x - crop_size * 0.5)
        top = max(0.0, center_y - crop_size * 0.5)
        right = min(float(image_width - 1), left + crop_size)
        bottom = min(float(image_height - 1), top + crop_size)
        crop_width = max(2.0, right - left)
        crop_height = max(2.0, bottom - top)
        return left, top, crop_width, crop_height

    def _photo_augment(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.85:
            image = TF.adjust_brightness(image, random.uniform(0.75, 1.25))
            image = TF.adjust_contrast(image, random.uniform(0.75, 1.25))
            image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))
            image = TF.adjust_hue(image, random.uniform(-0.05, 0.05))
        if random.random() < 0.20:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.4)))
        return image

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image_path, landmarks = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        landmarks = landmarks.clone()
        if self.use_full_image:
            crop_left, crop_top, crop_w, crop_h = 0.0, 0.0, float(image_width), float(image_height)
        else:
            crop_left, crop_top, crop_w, crop_h = self._sample_crop_box(landmarks, image_width, image_height)

        image = TF.resized_crop(
            image,
            top=int(round(crop_top)),
            left=int(round(crop_left)),
            height=max(2, int(round(crop_h))),
            width=max(2, int(round(crop_w))),
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        landmarks[:, 0] = (landmarks[:, 0] - crop_left) / crop_w
        landmarks[:, 1] = (landmarks[:, 1] - crop_top) / crop_h
        landmarks = landmarks.clamp(0.0, 1.0)

        if self.augment:
            image = self._photo_augment(image)

        tensor = TF.to_tensor(image)
        if self.augment and random.random() < 0.30:
            erase_size = random.randint(self.image_size // 10, self.image_size // 4)
            erase_x = random.randint(0, self.image_size - erase_size)
            erase_y = random.randint(0, self.image_size - erase_size)
            fill_value = torch.rand(3, 1, 1)
            tensor[:, erase_y : erase_y + erase_size, erase_x : erase_x + erase_size] = fill_value

        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return {
            "image": tensor,
            "landmarks": landmarks,
            "pose": torch.zeros(3, dtype=torch.float32),
        }


def _collect_samples(dataset_root: Path, folders: list[tuple[str, str | None]]) -> list[tuple[Path, Path]]:
    samples: list[tuple[Path, Path]] = []
    for parent_name, child_name in folders:
        folder = dataset_root / parent_name
        if child_name is not None:
            folder = folder / child_name
        for pts_path in sorted(folder.glob("*.pts")):
            image_path = None
            for extension in (".png", ".jpg", ".jpeg"):
                candidate = pts_path.with_suffix(extension)
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                continue
            samples.append((image_path, pts_path))
    if not samples:
        raise RuntimeError(f"No 300W samples were found under {dataset_root}")
    return samples


def _collect_samples_from_folder(folder: Path) -> list[tuple[Path, Path]]:
    samples: list[tuple[Path, Path]] = []
    for pts_path in sorted(folder.glob("*.pts")):
        image_path = None
        for extension in (".png", ".jpg", ".jpeg"):
            candidate = pts_path.with_suffix(extension)
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        samples.append((image_path, pts_path))
    return samples


class ThreeHundredWDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, Path]],
        *,
        image_size: int,
        augment: bool,
        max_samples: int | None,
        landmark_subset: str | None = None,
        crop_scale: float = 1.30,
        aug_scale_range: tuple[float, float] = (0.92, 1.18),
        aug_shift: float = 0.05,
        use_full_image: bool = False,
    ) -> None:
        self.samples = samples[:max_samples] if max_samples is not None else samples
        self.image_size = image_size
        self.augment = augment
        self.flip_order = FLIP_ORDER_68
        self.landmark_subset = landmark_subset
        self.crop_scale = crop_scale
        self.aug_scale_range = aug_scale_range
        self.aug_shift = aug_shift
        self.use_full_image = use_full_image


    def __len__(self) -> int:
        return len(self.samples)


    def _sample_crop_box(self, landmarks: Tensor, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        min_xy = landmarks.min(dim=0).values
        max_xy = landmarks.max(dim=0).values
        width, height = (max_xy - min_xy).tolist()
        center_x, center_y = ((min_xy + max_xy) * 0.5).tolist()
        scale = self.crop_scale
        shift_scale = 0.0
        if self.augment:
            scale *= random.uniform(*self.aug_scale_range)
            shift_scale = self.aug_shift
        crop_size = max(width, height) * scale
        center_x += random.uniform(-shift_scale, shift_scale) * crop_size
        center_y += random.uniform(-shift_scale, shift_scale) * crop_size

        left = max(0.0, center_x - crop_size * 0.5)
        top = max(0.0, center_y - crop_size * 0.5)
        right = min(float(image_width - 1), left + crop_size)
        bottom = min(float(image_height - 1), top + crop_size)
        crop_width = max(2.0, right - left)
        crop_height = max(2.0, bottom - top)
        return left, top, crop_width, crop_height


    def _augment_image(self, image: Image.Image, landmarks: Tensor) -> tuple[Image.Image, Tensor]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            landmarks = landmarks[self.flip_order]
            landmarks[:, 0] = 1.0 - landmarks[:, 0]

        if random.random() < 0.85:
            brightness = random.uniform(0.75, 1.25)
            contrast = random.uniform(0.75, 1.25)
            saturation = random.uniform(0.7, 1.3)
            hue = random.uniform(-0.05, 0.05)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)
            image = TF.adjust_hue(image, hue)

        if random.random() < 0.20:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.4)))

        return image, landmarks


    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image_path, pts_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        landmarks = _load_pts(pts_path)
        image_width, image_height = image.size
        if self.use_full_image:
            crop_left, crop_top, crop_width, crop_height = 0.0, 0.0, float(image_width), float(image_height)
        else:
            crop_left, crop_top, crop_width, crop_height = self._sample_crop_box(landmarks, image_width, image_height)

        image = TF.resized_crop(
            image,
            top=int(round(crop_top)),
            left=int(round(crop_left)),
            height=max(2, int(round(crop_height))),
            width=max(2, int(round(crop_width))),
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        landmarks = landmarks.clone()
        landmarks[:, 0] = (landmarks[:, 0] - crop_left) / crop_width
        landmarks[:, 1] = (landmarks[:, 1] - crop_top) / crop_height
        landmarks = landmarks.clamp(0.0, 1.0)

        if self.augment:
            image, landmarks = self._augment_image(image, landmarks)

        tensor = TF.to_tensor(image)

        if self.augment and random.random() < 0.30:
            erase_size = random.randint(self.image_size // 10, self.image_size // 4)
            erase_x = random.randint(0, self.image_size - erase_size)
            erase_y = random.randint(0, self.image_size - erase_size)
            fill_value = torch.rand(3, 1, 1)
            tensor[:, erase_y : erase_y + erase_size, erase_x : erase_x + erase_size] = fill_value

        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        landmarks = select_landmark_subset(landmarks, self.landmark_subset)
        return {
            "image": tensor,
            "landmarks": landmarks,
            "pose": torch.zeros(3, dtype=torch.float32),
        }


class WFLWAugmentedDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, Tensor, Tensor]],
        *,
        image_size: int,
        augment: bool,
        max_samples: int | None,
        landmark_subset: str | None = None,
        crop_scale: float = 1.30,
        aug_scale_range: tuple[float, float] = (0.94, 1.18),
        aug_shift: float = 0.04,
        use_full_image: bool = False,
    ) -> None:
        self.samples = samples[:max_samples] if max_samples is not None else samples
        self.image_size = image_size
        self.augment = augment
        self.flip_order = FLIP_ORDER_98
        self.landmark_subset = landmark_subset
        self.crop_scale = crop_scale
        self.aug_scale_range = aug_scale_range
        self.aug_shift = aug_shift
        self.use_full_image = use_full_image


    def __len__(self) -> int:
        return len(self.samples)


    def _sample_crop_box(self, landmarks: Tensor, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        min_xy = landmarks.min(dim=0).values
        max_xy = landmarks.max(dim=0).values
        width, height = (max_xy - min_xy).tolist()
        center_x, center_y = ((min_xy + max_xy) * 0.5).tolist()
        scale = self.crop_scale
        shift_scale = 0.0
        if self.augment:
            scale *= random.uniform(*self.aug_scale_range)
            shift_scale = self.aug_shift
        crop_size = max(width, height) * scale
        center_x += random.uniform(-shift_scale, shift_scale) * crop_size
        center_y += random.uniform(-shift_scale, shift_scale) * crop_size

        left = max(0.0, center_x - crop_size * 0.5)
        top = max(0.0, center_y - crop_size * 0.5)
        right = min(float(image_width - 1), left + crop_size)
        bottom = min(float(image_height - 1), top + crop_size)
        crop_width = max(2.0, right - left)
        crop_height = max(2.0, bottom - top)
        return left, top, crop_width, crop_height


    def _augment_image(self, image: Image.Image, landmarks: Tensor) -> tuple[Image.Image, Tensor]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            landmarks = landmarks[self.flip_order]
            landmarks[:, 0] = 1.0 - landmarks[:, 0]

        if random.random() < 0.85:
            brightness = random.uniform(0.75, 1.25)
            contrast = random.uniform(0.75, 1.25)
            saturation = random.uniform(0.7, 1.3)
            hue = random.uniform(-0.05, 0.05)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)
            image = TF.adjust_hue(image, hue)

        if random.random() < 0.20:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.4)))

        return image, landmarks


    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image_path, landmarks, pose = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        landmarks = landmarks.clone()
        landmarks[:, 0] *= image_width
        landmarks[:, 1] *= image_height
        if self.use_full_image:
            crop_left, crop_top, crop_width, crop_height = 0.0, 0.0, float(image_width), float(image_height)
        else:
            crop_left, crop_top, crop_width, crop_height = self._sample_crop_box(landmarks, image_width, image_height)

        image = TF.resized_crop(
            image,
            top=int(round(crop_top)),
            left=int(round(crop_left)),
            height=max(2, int(round(crop_height))),
            width=max(2, int(round(crop_width))),
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        landmarks[:, 0] = (landmarks[:, 0] - crop_left) / crop_width
        landmarks[:, 1] = (landmarks[:, 1] - crop_top) / crop_height
        landmarks = landmarks.clamp(0.0, 1.0)

        if self.augment:
            image, landmarks = self._augment_image(image, landmarks)

        tensor = TF.to_tensor(image)

        if self.augment and random.random() < 0.30:
            erase_size = random.randint(self.image_size // 10, self.image_size // 4)
            erase_x = random.randint(0, self.image_size - erase_size)
            erase_y = random.randint(0, self.image_size - erase_size)
            fill_value = torch.rand(3, 1, 1)
            tensor[:, erase_y : erase_y + erase_size, erase_x : erase_x + erase_size] = fill_value

        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        landmarks = select_landmark_subset(landmarks, self.landmark_subset)
        return {
            "image": tensor,
            "landmarks": landmarks,
            "pose": pose,
        }


class LaPaDataset(Dataset):
    """LaPa 106-point landmark dataset (https://github.com/JDAI-CV/lapa-dataset).

    Each sample is `(image_path, landmark_path)`; the landmark file's first line is the
    landmark count (106) and is followed by 106 lines of `x y` pixel coordinates inside
    the original image frame. We crop around the landmark bounding box (`crop_scale`)
    and produce coordinates normalized to `[0, 1]` inside that crop, matching the rest of
    the project's data contract. Horizontal flipping is disabled by default — the LaPa
    106-point flip permutation requires a verified table.
    """

    def __init__(
        self,
        samples: list[tuple[Path, Path]],
        *,
        image_size: int,
        augment: bool,
        max_samples: int | None,
        landmark_subset: str | None = None,
        crop_scale: float = 1.30,
        aug_scale_range: tuple[float, float] = (0.92, 1.18),
        aug_shift: float = 0.05,
        use_full_image: bool = False,
        enable_hflip: bool = False,
    ) -> None:
        self.samples = samples[:max_samples] if max_samples is not None else samples
        self.image_size = image_size
        self.augment = augment
        self.landmark_subset = landmark_subset
        self.crop_scale = crop_scale
        self.aug_scale_range = aug_scale_range
        self.aug_shift = aug_shift
        self.use_full_image = use_full_image
        self.enable_hflip = enable_hflip

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_crop_box(self, landmarks: Tensor, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        min_xy = landmarks.min(dim=0).values
        max_xy = landmarks.max(dim=0).values
        width, height = (max_xy - min_xy).tolist()
        center_x, center_y = ((min_xy + max_xy) * 0.5).tolist()
        scale = self.crop_scale
        shift_scale = 0.0
        if self.augment:
            scale *= random.uniform(*self.aug_scale_range)
            shift_scale = self.aug_shift
        crop_size = max(width, height) * scale
        center_x += random.uniform(-shift_scale, shift_scale) * crop_size
        center_y += random.uniform(-shift_scale, shift_scale) * crop_size
        left = max(0.0, center_x - crop_size * 0.5)
        top = max(0.0, center_y - crop_size * 0.5)
        right = min(float(image_width - 1), left + crop_size)
        bottom = min(float(image_height - 1), top + crop_size)
        crop_width = max(2.0, right - left)
        crop_height = max(2.0, bottom - top)
        return left, top, crop_width, crop_height

    def _photo_augment(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.85:
            image = TF.adjust_brightness(image, random.uniform(0.75, 1.25))
            image = TF.adjust_contrast(image, random.uniform(0.75, 1.25))
            image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))
            image = TF.adjust_hue(image, random.uniform(-0.05, 0.05))
        if random.random() < 0.20:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.4)))
        return image

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image_path, landmark_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        landmarks = _load_lapa_landmarks(landmark_path)
        image_width, image_height = image.size
        if self.use_full_image:
            crop_left, crop_top, crop_w, crop_h = 0.0, 0.0, float(image_width), float(image_height)
        else:
            crop_left, crop_top, crop_w, crop_h = self._sample_crop_box(landmarks, image_width, image_height)

        image = TF.resized_crop(
            image,
            top=int(round(crop_top)),
            left=int(round(crop_left)),
            height=max(2, int(round(crop_h))),
            width=max(2, int(round(crop_w))),
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        landmarks = landmarks.clone()
        landmarks[:, 0] = (landmarks[:, 0] - crop_left) / crop_w
        landmarks[:, 1] = (landmarks[:, 1] - crop_top) / crop_h
        landmarks = landmarks.clamp(0.0, 1.0)

        if self.augment:
            image = self._photo_augment(image)

        tensor = TF.to_tensor(image)
        if self.augment and random.random() < 0.30:
            erase_size = random.randint(self.image_size // 10, self.image_size // 4)
            erase_x = random.randint(0, self.image_size - erase_size)
            erase_y = random.randint(0, self.image_size - erase_size)
            fill_value = torch.rand(3, 1, 1)
            tensor[:, erase_y : erase_y + erase_size, erase_x : erase_x + erase_size] = fill_value

        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return {
            "image": tensor,
            "landmarks": landmarks,
            "pose": torch.zeros(3, dtype=torch.float32),
        }


class FaceSyntheticsZipDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, Tensor]],
        *,
        zip_path: str | Path,
        image_size: int,
        augment: bool,
        max_samples: int | None,
        landmark_subset: str | None = None,
        crop_scale: float = 1.15,
        aug_scale_range: tuple[float, float] = (0.95, 1.08),
        aug_shift: float = 0.02,
        use_full_image: bool = False,
    ) -> None:
        self.samples = samples[:max_samples] if max_samples is not None else samples
        self.zip_path = Path(zip_path)
        self.archive: zipfile.ZipFile | None = None
        self.image_size = image_size
        self.augment = augment
        self.landmark_subset = landmark_subset
        self.crop_scale = crop_scale
        self.aug_scale_range = aug_scale_range
        self.aug_shift = aug_shift
        self.use_full_image = use_full_image


    def __len__(self) -> int:
        return len(self.samples)


    def _get_archive(self) -> zipfile.ZipFile:
        if self.archive is None:
            self.archive = zipfile.ZipFile(self.zip_path)
        return self.archive


    def _sample_crop_box(self, landmarks: Tensor, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        min_xy = landmarks.min(dim=0).values
        max_xy = landmarks.max(dim=0).values
        width, height = (max_xy - min_xy).tolist()
        center_x, center_y = ((min_xy + max_xy) * 0.5).tolist()
        scale = self.crop_scale
        shift_scale = 0.0
        if self.augment:
            scale *= random.uniform(*self.aug_scale_range)
            shift_scale = self.aug_shift
        crop_size = max(width, height) * scale
        center_x += random.uniform(-shift_scale, shift_scale) * crop_size
        center_y += random.uniform(-shift_scale, shift_scale) * crop_size

        left = max(0.0, center_x - crop_size * 0.5)
        top = max(0.0, center_y - crop_size * 0.5)
        right = min(float(image_width - 1), left + crop_size)
        bottom = min(float(image_height - 1), top + crop_size)
        crop_width = max(2.0, right - left)
        crop_height = max(2.0, bottom - top)
        return left, top, crop_width, crop_height


    def _augment_image(self, image: Image.Image, landmarks: Tensor) -> tuple[Image.Image, Tensor]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            landmarks[:, 0] = 1.0 - landmarks[:, 0]

        if random.random() < 0.85:
            brightness = random.uniform(0.75, 1.25)
            contrast = random.uniform(0.75, 1.25)
            saturation = random.uniform(0.7, 1.3)
            hue = random.uniform(-0.05, 0.05)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)
            image = TF.adjust_hue(image, hue)

        if random.random() < 0.20:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.4)))
        return image, landmarks


    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image_name, landmarks = self.samples[index]
        archive = self._get_archive()
        with archive.open(image_name) as file:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image_width, image_height = image.size

        landmarks = landmarks.clone()
        landmarks[:, 0] /= image_width
        landmarks[:, 1] /= image_height
        landmarks = landmarks[:68]
        landmarks[:, 0] *= image_width
        landmarks[:, 1] *= image_height

        if self.use_full_image:
            crop_left, crop_top, crop_width, crop_height = 0.0, 0.0, float(image_width), float(image_height)
        else:
            crop_left, crop_top, crop_width, crop_height = self._sample_crop_box(landmarks, image_width, image_height)

        image = TF.resized_crop(
            image,
            top=int(round(crop_top)),
            left=int(round(crop_left)),
            height=max(2, int(round(crop_height))),
            width=max(2, int(round(crop_width))),
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        landmarks[:, 0] = (landmarks[:, 0] - crop_left) / crop_width
        landmarks[:, 1] = (landmarks[:, 1] - crop_top) / crop_height
        landmarks = landmarks.clamp(0.0, 1.0)

        if self.augment:
            image, landmarks = self._augment_image(image, landmarks)

        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        landmarks = select_landmark_subset(landmarks, self.landmark_subset)
        return {
            "image": tensor,
            "landmarks": landmarks,
            "pose": torch.zeros(3, dtype=torch.float32),
        }


class PseudoCelebADataset(Dataset):
    """CelebA images with FAN-teacher 68-point pseudo labels.

    `samples` items are tuples ``(image_path, landmarks_in_crop, (left, top, side))`` where
    ``landmarks_in_crop`` is normalized to ``[0, 1]`` inside the FAN-time center crop
    ``(left, top, left+side, top+side)`` of the original CelebA image.
    """

    def __init__(
        self,
        samples: list[tuple[Path, Tensor, tuple[float, float, float]]],
        *,
        image_size: int,
        augment: bool,
        max_samples: int | None,
        landmark_subset: str | None = None,
        crop_scale: float = 1.30,
        aug_scale_range: tuple[float, float] = (0.92, 1.18),
        aug_shift: float = 0.05,
        use_full_image: bool = False,
    ) -> None:
        self.samples = samples[:max_samples] if max_samples is not None else samples
        self.image_size = image_size
        self.augment = augment
        self.flip_order = FLIP_ORDER_68
        self.landmark_subset = landmark_subset
        self.crop_scale = crop_scale
        self.aug_scale_range = aug_scale_range
        self.aug_shift = aug_shift
        self.use_full_image = use_full_image

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_crop_box(self, landmarks: Tensor, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        min_xy = landmarks.min(dim=0).values
        max_xy = landmarks.max(dim=0).values
        width, height = (max_xy - min_xy).tolist()
        center_x, center_y = ((min_xy + max_xy) * 0.5).tolist()
        scale = self.crop_scale
        shift_scale = 0.0
        if self.augment:
            scale *= random.uniform(*self.aug_scale_range)
            shift_scale = self.aug_shift
        crop_size = max(width, height) * scale
        center_x += random.uniform(-shift_scale, shift_scale) * crop_size
        center_y += random.uniform(-shift_scale, shift_scale) * crop_size

        left = max(0.0, center_x - crop_size * 0.5)
        top = max(0.0, center_y - crop_size * 0.5)
        right = min(float(image_width - 1), left + crop_size)
        bottom = min(float(image_height - 1), top + crop_size)
        crop_width = max(2.0, right - left)
        crop_height = max(2.0, bottom - top)
        return left, top, crop_width, crop_height

    def _augment_image(self, image: Image.Image, landmarks: Tensor) -> tuple[Image.Image, Tensor]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            landmarks = landmarks[self.flip_order]
            landmarks[:, 0] = 1.0 - landmarks[:, 0]
        if random.random() < 0.85:
            image = TF.adjust_brightness(image, random.uniform(0.75, 1.25))
            image = TF.adjust_contrast(image, random.uniform(0.75, 1.25))
            image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))
            image = TF.adjust_hue(image, random.uniform(-0.05, 0.05))
        if random.random() < 0.20:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.4)))
        return image, landmarks

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image_path, landmarks_in_crop, crop_xyz = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        crop_left, crop_top, crop_side = (float(value) for value in crop_xyz)
        # Re-project pseudo landmarks (normalized within the FAN center crop) back to original-image pixel coords.
        landmarks = landmarks_in_crop.clone().to(torch.float32)
        landmarks[:, 0] = landmarks[:, 0] * crop_side + crop_left
        landmarks[:, 1] = landmarks[:, 1] * crop_side + crop_top

        if self.use_full_image:
            sample_left, sample_top, sample_w, sample_h = 0.0, 0.0, float(original_width), float(original_height)
        else:
            sample_left, sample_top, sample_w, sample_h = self._sample_crop_box(landmarks, original_width, original_height)

        image = TF.resized_crop(
            image,
            top=int(round(sample_top)),
            left=int(round(sample_left)),
            height=max(2, int(round(sample_h))),
            width=max(2, int(round(sample_w))),
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        landmarks[:, 0] = (landmarks[:, 0] - sample_left) / sample_w
        landmarks[:, 1] = (landmarks[:, 1] - sample_top) / sample_h
        landmarks = landmarks.clamp(0.0, 1.0)

        if self.augment:
            image, landmarks = self._augment_image(image, landmarks)

        tensor = TF.to_tensor(image)

        if self.augment and random.random() < 0.30:
            erase_size = random.randint(self.image_size // 10, self.image_size // 4)
            erase_x = random.randint(0, self.image_size - erase_size)
            erase_y = random.randint(0, self.image_size - erase_size)
            fill_value = torch.rand(3, 1, 1)
            tensor[:, erase_y : erase_y + erase_size, erase_x : erase_x + erase_size] = fill_value

        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        landmarks = select_landmark_subset(landmarks, self.landmark_subset)
        return {
            "image": tensor,
            "landmarks": landmarks,
            "pose": torch.zeros(3, dtype=torch.float32),
        }


def _load_celeba_pseudo_samples(npz_path: Path, image_dir: Path) -> list[tuple[Path, Tensor, tuple[float, float, float]]]:
    import numpy as np
    payload = np.load(npz_path, allow_pickle=False)
    names = payload["names"]
    landmarks_array = payload["landmarks"]
    crops_array = payload["crops"]
    samples: list[tuple[Path, Tensor, tuple[float, float, float]]] = []
    for name, lmk, crop in zip(names, landmarks_array, crops_array):
        path = image_dir / str(name)
        if not path.exists():
            continue
        samples.append((path, torch.from_numpy(lmk.astype("float32")), tuple(float(v) for v in crop)))
    return samples


def select_landmark_subset(landmarks: Tensor, landmark_subset: str | None) -> Tensor:
    if landmark_subset is None:
        return landmarks
    if landmark_subset == "wflw68_pfld":
        jaw = landmarks[0:33:2]
        left_brow = landmarks[33:38]
        right_brow = landmarks[42:47]
        nose = landmarks[51:61]
        left_eye = torch.stack(
            [
                0.5 * (landmarks[60] + landmarks[62]),
                0.5 * (landmarks[62] + landmarks[64]),
                landmarks[64],
                0.5 * (landmarks[64] + landmarks[66]),
                0.5 * (landmarks[60] + landmarks[66]),
            ],
            dim=0,
        )
        right_eye = torch.stack(
            [
                landmarks[68],
                0.5 * (landmarks[68] + landmarks[70]),
                0.5 * (landmarks[70] + landmarks[72]),
                landmarks[72],
                0.5 * (landmarks[72] + landmarks[74]),
                0.5 * (landmarks[68] + landmarks[74]),
            ],
            dim=0,
        )
        mouth = landmarks[76:96]
        return torch.cat([jaw, left_brow, right_brow, nose, left_eye, right_eye, mouth], dim=0)
    if landmark_subset == "wflw68":
        subset_indices = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
            33, 34, 35, 36, 37,
            42, 43, 44, 45, 46,
            51, 52, 53, 54, 55, 56, 57, 58, 59,
            60, 61, 63, 64, 65, 67,
            68, 69, 71, 72, 73, 75,
            76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
        ]
        return landmarks[subset_indices]
    if landmark_subset == "wflw5":
        left_eye = landmarks[60:68].mean(dim=0)
        right_eye = landmarks[68:76].mean(dim=0)
        nose_tip = landmarks[54]
        mouth_left = landmarks[76]
        mouth_right = landmarks[82]
        return torch.stack([left_eye, right_eye, nose_tip, mouth_left, mouth_right], dim=0)
    raise ValueError(f"Unsupported landmark subset: {landmark_subset}")


def create_dataloaders(config: dict) -> DataBundle:
    data_config = config["data"]
    image_size = int(data_config["image_size"])
    root = data_config["root"]
    download = bool(data_config["download"])
    num_workers = int(data_config["num_workers"])
    if os.name == "nt":
        num_workers = min(num_workers, 4)
    pin_memory = torch.cuda.is_available()
    landmark_subset = data_config.get("landmark_subset")
    overfit_same_split = bool(data_config.get("overfit_same_split", False))

    dataset_name = data_config.get("dataset", "300w")
    if dataset_name == "facesynthetics":
        all_samples = _load_facesynthetics_zip(data_config["zip_path"], max_samples=data_config.get("max_samples"))
        split_generator = random.Random(int(config["seed"]))
        split_generator.shuffle(all_samples)
        valid_ratio = float(data_config["valid_ratio"])
        test_ratio = float(data_config.get("test_ratio", valid_ratio))
        valid_count = max(1, int(len(all_samples) * valid_ratio))
        test_count = max(1, int(len(all_samples) * test_ratio))
        valid_samples = all_samples[:valid_count]
        test_samples = all_samples[valid_count : valid_count + test_count]
        train_samples = all_samples[valid_count + test_count :]
        dataset_cls = FaceSyntheticsZipDataset
    elif dataset_name == "wflw_augmented":
        train_dir, test_dir = _prepare_wflw_augmented(
            root,
            download=download,
            train_url=data_config["train_archive_url"],
            test_url=data_config["test_archive_url"],
        )
        train_samples = _load_wflw_rows(train_dir)
        test_samples = _load_wflw_rows(test_dir)
        dataset_cls: type[Dataset] = WFLWAugmentedDataset
    elif dataset_name == "lapa":
        lapa_root = Path(data_config.get("lapa_root", Path(root) / "LaPa"))
        train_samples = _collect_lapa_samples(lapa_root / "train")
        val_samples_native = _collect_lapa_samples(lapa_root / "val")
        test_samples = _collect_lapa_samples(lapa_root / "test")
        dataset_cls: type[Dataset] = LaPaDataset
    elif dataset_name == "lapa_mixed":
        # LaPa + JD-landmark + Pseudo-WFLW mixed training
        lapa_root = Path(data_config.get("lapa_root", Path(root) / "LaPa"))
        train_samples = _collect_lapa_samples(lapa_root / "train")
        val_samples_native = _collect_lapa_samples(lapa_root / "val")
        test_samples = _collect_lapa_samples(lapa_root / "test")
        dataset_cls: type[Dataset] = LaPaDataset
    elif dataset_name == "jd_landmark":
        # JD-landmark (ICME 2021 FLL3) dataset - same 106-point format as LaPa
        jd_root = Path(data_config["lapa_root"])
        jd_samples = _collect_jd_landmark_samples(jd_root)
        rng = random.Random(int(config["seed"]))
        rng.shuffle(jd_samples)
        valid_ratio = float(data_config.get("valid_ratio", 0.1))
        test_ratio = float(data_config.get("test_ratio", 0.1))
        valid_count = max(1, int(len(jd_samples) * valid_ratio))
        test_count = max(1, int(len(jd_samples) * test_ratio))
        val_samples_native = jd_samples[:valid_count]
        test_samples = jd_samples[valid_count : valid_count + test_count]
        train_samples = jd_samples[valid_count + test_count :]
        dataset_cls: type[Dataset] = LaPaDataset
    else:
        dataset_root = _prepare_300w(root, download=download, archive_url=data_config["archive_url"])
        if (dataset_root / "01_Indoor").exists() and (dataset_root / "02_Outdoor").exists():
            split_strategy = data_config.get("split_strategy", "indoor_outdoor")
            if split_strategy == "all_random":
                train_samples = _collect_samples_from_folder(dataset_root / "01_Indoor")
                train_samples.extend(_collect_samples_from_folder(dataset_root / "02_Outdoor"))
                test_samples = []
            else:
                train_samples = _collect_samples_from_folder(dataset_root / "01_Indoor")
                test_samples = _collect_samples_from_folder(dataset_root / "02_Outdoor")
        else:
            train_samples = _collect_samples(dataset_root, TRAIN_FOLDERS)
            test_samples = _collect_samples(dataset_root, TEST_FOLDERS)
        dataset_cls = ThreeHundredWDataset

    if dataset_name != "facesynthetics":
        split_generator = random.Random(int(config["seed"]))
        split_generator.shuffle(train_samples)
        if dataset_name in ("lapa", "lapa_mixed", "jd_landmark"):
            valid_samples = val_samples_native
        else:
            valid_ratio = float(data_config["valid_ratio"])
            valid_count = max(1, int(len(train_samples) * valid_ratio))
            valid_samples = train_samples[:valid_count]
            train_samples = train_samples[valid_count:]

            if not test_samples:
                test_ratio = float(data_config.get("test_ratio", valid_ratio))
                test_count = max(1, int(len(train_samples) * test_ratio))
                test_samples = train_samples[:test_count]
                train_samples = train_samples[test_count:]

    if overfit_same_split:
        shared_count = int(data_config.get("train_subset") or len(train_samples))
        shared_samples = train_samples[:shared_count]
        train_samples = shared_samples
        valid_samples = shared_samples
        test_samples = shared_samples

    common_dataset_kwargs = {
        "image_size": image_size,
        "landmark_subset": landmark_subset,
    }
    if dataset_name == "facesynthetics":
        common_dataset_kwargs["zip_path"] = data_config["zip_path"]

    train_dataset = dataset_cls(
        train_samples,
        augment=bool(data_config.get("train_augment", True)),
        max_samples=data_config.get("train_subset"),
        **common_dataset_kwargs,
    )
    valid_dataset = dataset_cls(
        valid_samples,
        augment=False,
        max_samples=data_config.get("valid_subset"),
        **common_dataset_kwargs,
    )
    test_dataset = dataset_cls(
        test_samples,
        augment=False,
        max_samples=data_config.get("test_subset"),
        **common_dataset_kwargs,
    )

    crop_scale = float(data_config.get("crop_scale", 1.30))
    aug_scale_range = tuple(float(value) for value in data_config.get("aug_scale_range", [0.92, 1.18]))
    aug_shift = float(data_config.get("aug_shift", 0.05))
    use_full_image = bool(data_config.get("use_full_image", False))
    train_dataset.crop_scale = crop_scale
    train_dataset.aug_scale_range = aug_scale_range
    train_dataset.aug_shift = aug_shift
    train_dataset.use_full_image = use_full_image
    valid_dataset.crop_scale = crop_scale
    valid_dataset.aug_scale_range = aug_scale_range
    valid_dataset.aug_shift = aug_shift
    valid_dataset.use_full_image = use_full_image
    test_dataset.crop_scale = crop_scale
    test_dataset.aug_scale_range = aug_scale_range
    test_dataset.aug_shift = aug_shift
    test_dataset.use_full_image = use_full_image

    # ---- lapa_mixed: merge JD-landmark + Pseudo-WFLW into LaPa training set ----
    sampler = None
    sampler_replacement = True
    if dataset_name == "lapa_mixed":
        mixed_cfg = data_config.get("mixed", {})
        datasets_in_order: list[Dataset] = [train_dataset]
        ranges: list[tuple[str, int]] = [("lapa", len(train_dataset))]

        # Add JD-landmark (ICME 2021 FLL3)
        jd_root = Path(mixed_cfg.get("jd_root", Path(root) / "jd_landmark" / "FLL3_dataset"))
        if jd_root.exists():
            jd_samples = _collect_jd_landmark_samples(jd_root)
            max_jd = mixed_cfg.get("max_jd_samples")
            if max_jd:
                jd_samples = jd_samples[:int(max_jd)]
            jd_dataset = LaPaDataset(
                jd_samples,
                image_size=image_size,
                augment=bool(data_config.get("train_augment", True)),
                max_samples=None,
                landmark_subset=landmark_subset,
                crop_scale=crop_scale,
                aug_scale_range=aug_scale_range,
                aug_shift=aug_shift,
                enable_hflip=bool(data_config.get("enable_hflip", False)),
            )
            datasets_in_order.append(jd_dataset)
            ranges.append(("jd_landmark", len(jd_dataset)))

        # Add Pseudo-WFLW (106-point pseudo labels from HRNet teacher)
        pseudo_wflw_csv = Path(mixed_cfg.get("pseudo_wflw_csv", Path(root) / "wflw_pseudo_106" / "train_data.csv"))
        if pseudo_wflw_csv.exists():
            pw_samples = _load_pseudo_wflw_rows(pseudo_wflw_csv)
            max_pw = mixed_cfg.get("max_pseudo_wflw_samples")
            if max_pw:
                pw_samples = pw_samples[:int(max_pw)]
            pw_dataset = PseudoWFLWDataset(
                pw_samples,
                image_size=image_size,
                augment=bool(data_config.get("train_augment", True)),
                max_samples=None,
                landmark_subset=landmark_subset,
                crop_scale=crop_scale,
                aug_scale_range=aug_scale_range,
                aug_shift=aug_shift,
                enable_hflip=bool(data_config.get("enable_hflip", False)),
            )
            datasets_in_order.append(pw_dataset)
            ranges.append(("pseudo_wflw", len(pw_dataset)))

        train_dataset = ConcatDataset(datasets_in_order)
        share_summary = ", ".join(f"{name}={count}" for name, count in ranges)
        print(f"[data] mixed training: {share_summary}, total={len(train_dataset)}")

    pseudo_config = data_config.get("pseudo_celeba")
    wflw_aug_config = data_config.get("wflw_aug")
    sampler = None
    sampler_replacement = True
    if pseudo_config or wflw_aug_config:
        real_300w_count = len(train_dataset)
        datasets_in_order: list[Dataset] = [train_dataset]
        ranges: list[tuple[str, int]] = [("real_300w", real_300w_count)]

        wflw_count = 0
        if wflw_aug_config:
            wflw_train_dir = Path(wflw_aug_config["train_dir"]) if "train_dir" in wflw_aug_config else None
            if wflw_train_dir is None:
                wflw_train_dir, _ = _prepare_wflw_augmented(
                    root,
                    download=download,
                    train_url=wflw_aug_config["train_archive_url"],
                    test_url=wflw_aug_config["test_archive_url"],
                )
            wflw_samples = _load_wflw_rows(wflw_train_dir)
            max_wflw = wflw_aug_config.get("max_samples")
            if max_wflw:
                wflw_samples = wflw_samples[: int(max_wflw)]
            wflw_dataset = WFLWAugmentedDataset(
                wflw_samples,
                image_size=image_size,
                augment=bool(data_config.get("train_augment", True)),
                max_samples=None,
                landmark_subset=str(wflw_aug_config.get("landmark_subset", "wflw68")),
                crop_scale=float(wflw_aug_config.get("crop_scale", crop_scale)),
                aug_scale_range=tuple(float(v) for v in wflw_aug_config.get("aug_scale_range", aug_scale_range)),
                aug_shift=float(wflw_aug_config.get("aug_shift", aug_shift)),
                use_full_image=False,
            )
            datasets_in_order.append(wflw_dataset)
            wflw_count = len(wflw_dataset)
            ranges.append(("real_wflw", wflw_count))

        pseudo_count = 0
        if pseudo_config:
            npz_path = Path(pseudo_config["npz_path"])
            image_dir = Path(pseudo_config["image_dir"])
            pseudo_samples = _load_celeba_pseudo_samples(npz_path, image_dir)
            max_pseudo = pseudo_config.get("max_samples")
            if max_pseudo:
                pseudo_samples = pseudo_samples[: int(max_pseudo)]
            pseudo_dataset = PseudoCelebADataset(
                pseudo_samples,
                image_size=image_size,
                augment=bool(data_config.get("train_augment", True)),
                max_samples=None,
                landmark_subset=landmark_subset,
                crop_scale=float(pseudo_config.get("crop_scale", crop_scale)),
                aug_scale_range=tuple(float(v) for v in pseudo_config.get("aug_scale_range", aug_scale_range)),
                aug_shift=float(pseudo_config.get("aug_shift", aug_shift)),
                use_full_image=False,
            )
            datasets_in_order.append(pseudo_dataset)
            pseudo_count = len(pseudo_dataset)
            ranges.append(("pseudo", pseudo_count))

        train_dataset = ConcatDataset(datasets_in_order)

        # Sampling ratios: real_ratio (real total) vs pseudo, plus split inside real for 300W vs WFLW.
        primary_cfg = pseudo_config if pseudo_config else wflw_aug_config
        real_ratio = float(primary_cfg.get("real_ratio", 0.5))
        if pseudo_count == 0:
            real_ratio = 1.0
        real_ratio = min(max(real_ratio, 1e-3), 1.0)
        wflw_within_real = float((wflw_aug_config or {}).get("wflw_within_real", 0.85)) if wflw_count else 0.0
        wflw_within_real = min(max(wflw_within_real, 0.0), 1.0)

        share_300w = real_ratio * (1.0 - wflw_within_real) if wflw_count else real_ratio
        share_wflw = real_ratio * wflw_within_real if wflw_count else 0.0
        share_pseudo = 1.0 - real_ratio if pseudo_count else 0.0

        weights_segments: list[tuple[float, int]] = [
            (share_300w / max(real_300w_count, 1), real_300w_count),
        ]
        if wflw_count:
            weights_segments.append((share_wflw / max(wflw_count, 1), wflw_count))
        if pseudo_count:
            weights_segments.append((share_pseudo / max(pseudo_count, 1), pseudo_count))

        weight_values: list[float] = []
        for per_sample_weight, count in weights_segments:
            weight_values.extend([per_sample_weight] * count)
        weights = torch.tensor(weight_values, dtype=torch.double)

        epoch_steps = int(primary_cfg.get("epoch_steps", 0))
        if epoch_steps > 0:
            num_samples = epoch_steps * int(config["train"]["batch_size"])
        else:
            num_samples = max(real_300w_count, int(real_300w_count / max(share_300w, 1e-3)))
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)

        share_summary = ", ".join(
            f"{name}={count}" for name, count in ranges
        )
        print(
            f"[data] joined extra: {share_summary}; share_300w={share_300w:.3f} "
            f"share_wflw={share_wflw:.3f} share_pseudo={share_pseudo:.3f} "
            f"num_samples_per_epoch={num_samples}"
        )

    batch_size = int(config["train"]["batch_size"])
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if sampler is None:
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=True, **loader_kwargs)
    return DataBundle(
        train=train_loader,
        valid=DataLoader(valid_dataset, shuffle=False, drop_last=False, **loader_kwargs),
        test=DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs),
    )
