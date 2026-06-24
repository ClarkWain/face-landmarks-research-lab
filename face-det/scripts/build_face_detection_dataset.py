from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
OUT_ROOT = ROOT / "face-det" / "data" / "yolo_face_det"


def ensure_dirs() -> None:
    for split in ("train", "val", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yolo_label(label_path: Path, box: tuple[float, float, float, float], width: int, height: int) -> None:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    label_path.write_text(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8")


def expand_box_from_points(points: list[tuple[float, float]], width: int, height: int, scale: float) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(x2 - x1, y2 - y1) * scale
    left = max(0.0, cx - side * 0.5)
    top = max(0.0, cy - side * 0.5)
    right = min(float(width - 1), cx + side * 0.5)
    bottom = min(float(height - 1), cy + side * 0.5)
    return left, top, right, bottom


def copy_with_label(image_path: Path, split: str, source_name: str, box: tuple[float, float, float, float]) -> None:
    target_stem = f"{source_name}__{image_path.stem}"
    target_image = OUT_ROOT / "images" / split / f"{target_stem}{image_path.suffix}"
    target_label = OUT_ROOT / "labels" / split / f"{target_stem}.txt"
    shutil.copy2(image_path, target_image)
    with Image.open(image_path) as image:
        width, height = image.size
    write_yolo_label(target_label, box, width, height)


def add_lapa(scale: float = 1.35, include_test: bool = False) -> int:
    count = 0
    mapping = [("train", "train"), ("val", "val")]
    if include_test:
        mapping.append(("test", "test"))
    for split_name, out_split in mapping:
        image_dir = DATA_ROOT / "LaPa" / split_name / "images"
        landmark_dir = DATA_ROOT / "LaPa" / split_name / "landmarks"
        for landmark_path in sorted(landmark_dir.glob("*.txt")):
            image_path = image_dir / f"{landmark_path.stem}.jpg"
            if not image_path.exists():
                continue
            lines = [line.strip() for line in landmark_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            total = int(lines[0])
            points = [tuple(float(v) for v in line.split()[:2]) for line in lines[1 : total + 1]]
            with Image.open(image_path) as image:
                width, height = image.size
            box = expand_box_from_points(points, width, height, scale)
            copy_with_label(image_path, out_split, "lapa", box)
            count += 1
    return count


def add_jd(scale: float = 1.35) -> int:
    # The current workspace contains JD images but not a stable, directly-consumable
    # landmark CSV/index for them. We skip JD in the first detector baseline to keep
    # the dataset builder deterministic and fast.
    return 0


def add_wflw(scale: float = 1.35, include: bool = True) -> int:
    if not include:
        return 0
    count = 0
    for split_name, out_split in (("train_data", "train"), ("test_data", "val")):
        image_dir = DATA_ROOT / "wflw_extracted" / split_name / "imgs"
        labels_path = DATA_ROOT / "wflw_extracted" / split_name / "labels.csv"
        with labels_path.open("r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) != 206:
                    continue
                image_path = image_dir / Path(row[-1]).name
                if not image_path.exists():
                    continue
                coords = [float(v) for v in row[:196]]
                with Image.open(image_path) as image:
                    width, height = image.size
                points = []
                for i in range(0, 196, 2):
                    points.append((coords[i] * width, coords[i + 1] * height))
                box = expand_box_from_points(points, width, height, scale)
                copy_with_label(image_path, out_split, "wflw", box)
                count += 1
    return count


def add_testdata1_benchmark() -> int:
    count = 0
    image_dir = DATA_ROOT / "Test_data1" / "picture"
    rect_dir = DATA_ROOT / "Test_data1" / "rect"
    for rect_path in sorted(rect_dir.glob("*.rect")):
        image_path = image_dir / rect_path.name.replace(".rect", "")
        if not image_path.exists():
            continue
        vals = [float(v) for v in rect_path.read_text(encoding="utf-8").split()[:4]]
        box = tuple(vals)  # x1 y1 x2 y2
        copy_with_label(image_path, "test", "testdata1", box)
        count += 1
    return count


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["full", "lapa_only"], default="full")
    args = parser.parse_args()

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    ensure_dirs()

    counts = {}
    counts["lapa"] = add_lapa(include_test=False)
    counts["jd_landmark"] = add_jd()
    counts["wflw"] = add_wflw(include=args.profile == "full")
    counts["test_data1"] = add_testdata1_benchmark()
    counts["profile"] = args.profile

    for split in ("train", "val", "test"):
        image_count = len(list((OUT_ROOT / "images" / split).glob("*")))
        label_count = len(list((OUT_ROOT / "labels" / split).glob("*.txt")))
        print(f"{split}: images={image_count} labels={label_count}")
    print(counts)


if __name__ == "__main__":
    main()
