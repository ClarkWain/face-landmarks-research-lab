from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image
from torchvision.datasets import WIDERFace


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
OUT_ROOT = ROOT / "face-det" / "data" / "yolo_face_det"
RNG = random.Random(3407)


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


def append_yolo_label(label_path: Path, box: tuple[float, float, float, float], width: int, height: int) -> None:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    with label_path.open("a", encoding="utf-8") as f:
        f.write(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n")


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


def add_widerface(split: str, out_split: str) -> int:
    dataset = WIDERFace(DATA_ROOT / "WIDERFace", split=split, download=False)
    count = 0
    for index in range(len(dataset)):
        image, target = dataset[index]
        image_path = Path(dataset.img_info[index]["img_path"])  # type: ignore[index]
        if target is None:
            continue
        bbox_tensor = target["bbox"]
        invalid = target["invalid"]
        width, height = image.size
        target_stem = f"wider_{split}__{image_path.stem}"
        target_image = OUT_ROOT / "images" / out_split / f"{target_stem}{image_path.suffix}"
        target_label = OUT_ROOT / "labels" / out_split / f"{target_stem}.txt"
        shutil.copy2(image_path, target_image)
        target_label.write_text("", encoding="utf-8")
        for bbox, inv in zip(bbox_tensor.tolist(), invalid.tolist()):
            if int(inv) != 0:
                continue
            x, y, w, h = bbox
            if w <= 1 or h <= 1:
                continue
            append_yolo_label(target_label, (x, y, x + w, y + h), width, height)
        count += 1
    return count


def add_crowdhuman(crowdhuman_root: Path) -> int:
    images_dir = crowdhuman_root / "Images"
    ann_path = crowdhuman_root / "annotation_train.odgt"
    if not images_dir.exists() or not ann_path.exists():
        return 0

    count = 0
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            image_path = images_dir / f"{record['ID']}.jpg"
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            target_stem = f"crowdhuman__{image_path.stem}"
            target_image = OUT_ROOT / "images" / "train" / f"{target_stem}{image_path.suffix}"
            target_label = OUT_ROOT / "labels" / "train" / f"{target_stem}.txt"
            shutil.copy2(image_path, target_image)
            target_label.write_text("", encoding="utf-8")
            for gtbox in record.get("gtboxes", []):
                if gtbox.get("tag") != "person":
                    continue
                fbox = gtbox.get("fbox")
                if not fbox or len(fbox) != 4:
                    continue
                x, y, w, h = [float(v) for v in fbox]
                if w <= 4 or h <= 4:
                    continue
                append_yolo_label(target_label, (x, y, x + w, y + h), width, height)
            count += 1
    return count


def _lapa_train_samples() -> list[tuple[Path, list[tuple[float, float]], tuple[float, float, float, float]]]:
    samples: list[tuple[Path, list[tuple[float, float]], tuple[float, float, float, float]]] = []
    image_dir = DATA_ROOT / "LaPa" / "train" / "images"
    landmark_dir = DATA_ROOT / "LaPa" / "train" / "landmarks"
    for landmark_path in sorted(landmark_dir.glob("*.txt")):
        image_path = image_dir / f"{landmark_path.stem}.jpg"
        if not image_path.exists():
            continue
        lines = [line.strip() for line in landmark_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        total = int(lines[0])
        points = [tuple(float(v) for v in line.split()[:2]) for line in lines[1 : total + 1]]
        with Image.open(image_path) as image:
            width, height = image.size
        box = expand_box_from_points(points, width, height, 1.35)
        samples.append((image_path, points, box))
    return samples


def add_synthetic_small_faces(num_train: int = 0, num_val: int = 0) -> dict[str, int]:
    pool = _lapa_train_samples()
    stats = {"train": 0, "val": 0}
    if not pool:
        return stats

    def render_one(index: int, split: str) -> None:
        background_path, _, _ = RNG.choice(pool)
        with Image.open(background_path).convert("RGB") as bg_image:
            canvas = bg_image.resize((640, 640), Image.BILINEAR)
        label_path = OUT_ROOT / "labels" / split / f"synthetic_small__{index:06d}.txt"
        image_path = OUT_ROOT / "images" / split / f"synthetic_small__{index:06d}.jpg"
        label_path.write_text("", encoding="utf-8")

        faces_per_image = RNG.randint(2, 5)
        placed: list[tuple[int, int, int, int]] = []
        for _ in range(faces_per_image):
            src_path, _, src_box = RNG.choice(pool)
            with Image.open(src_path).convert("RGB") as src_img:
                crop = src_img.crop((int(src_box[0]), int(src_box[1]), int(src_box[2]), int(src_box[3])))
                side = RNG.randint(12, 48)
                crop = crop.resize((side, side), Image.BILINEAR)

            placed_ok = False
            for _try in range(30):
                x1 = RNG.randint(0, 640 - side)
                y1 = RNG.randint(0, 640 - side)
                x2 = x1 + side
                y2 = y1 + side
                overlaps = False
                for ox1, oy1, ox2, oy2 in placed:
                    inter_w = max(0, min(x2, ox2) - max(x1, ox1))
                    inter_h = max(0, min(y2, oy2) - max(y1, oy1))
                    if inter_w * inter_h > 0:
                        overlaps = True
                        break
                if overlaps:
                    continue
                canvas.paste(crop, (x1, y1))
                append_yolo_label(label_path, (x1, y1, x2, y2), 640, 640)
                placed.append((x1, y1, x2, y2))
                placed_ok = True
                break
            if not placed_ok:
                continue

        canvas.save(image_path, quality=95)

    for i in range(num_train):
        render_one(i, "train")
        stats["train"] += 1
    for i in range(num_val):
        render_one(i, "val")
        stats["val"] += 1
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["full", "lapa_only", "robust_mix"], default="full")
    parser.add_argument("--synthetic-train", type=int, default=0)
    parser.add_argument("--synthetic-val", type=int, default=0)
    parser.add_argument("--crowdhuman-root", default=None)
    args = parser.parse_args()

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    ensure_dirs()

    counts = {}
    counts["lapa"] = add_lapa(include_test=False)
    counts["jd_landmark"] = add_jd()
    counts["wflw"] = add_wflw(include=args.profile in {"full", "robust_mix"})
    if args.profile in {"full", "robust_mix"} and (DATA_ROOT / "WIDERFace" / "widerface").exists():
        try:
            counts["wider_train"] = add_widerface("train", "train")
            counts["wider_val"] = add_widerface("val", "val")
        except Exception as exc:  # noqa: BLE001
            print(f"skip widerface: {exc}")
            counts["wider_train"] = 0
            counts["wider_val"] = 0
    else:
        counts["wider_train"] = 0
        counts["wider_val"] = 0

    if args.crowdhuman_root:
        counts["crowdhuman"] = add_crowdhuman(Path(args.crowdhuman_root))
    else:
        counts["crowdhuman"] = 0

    synth_stats = add_synthetic_small_faces(args.synthetic_train, args.synthetic_val)
    counts["synthetic_train"] = synth_stats["train"]
    counts["synthetic_val"] = synth_stats["val"]
    counts["test_data1"] = add_testdata1_benchmark()
    counts["profile"] = args.profile

    for split in ("train", "val", "test"):
        image_count = len(list((OUT_ROOT / "images" / split).glob("*")))
        label_count = len(list((OUT_ROOT / "labels" / split).glob("*.txt")))
        print(f"{split}: images={image_count} labels={label_count}")
    print(counts)


if __name__ == "__main__":
    main()
