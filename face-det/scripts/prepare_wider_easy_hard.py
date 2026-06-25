from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
WIDER_ROOT = ROOT / "data" / "WIDERFace" / "widerface"


def _mat_root() -> Path:
    candidate = WIDER_ROOT / "eval_tools" / "ground_truth"
    if candidate.exists():
        return candidate
    return WIDER_ROOT / "wider_face_split"


def ensure_split_dirs(base: Path) -> None:
    (base / "images" / "val").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "val").mkdir(parents=True, exist_ok=True)


def write_yolo_label(label_path: Path, boxes: list[tuple[float, float, float, float]], width: int, height: int) -> None:
    with label_path.open("w", encoding="utf-8") as f:
        for x, y, w, h in boxes:
            cx = (x + w * 0.5) / width
            cy = (y + h * 0.5) / height
            bw = w / width
            bh = h / height
            f.write(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n")


def _to_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "item"):
        try:
            return str(obj.item())
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        data = obj.tolist()
        if isinstance(data, list) and len(data) == 1:
            return _to_text(data[0])
        return str(data)
    return str(obj)


def _unwrap_singleton_object_array(obj):
    while hasattr(obj, "dtype") and obj.dtype == object and obj.size == 1:
        obj = obj.flat[0]
    return obj


def build_subset(level: str) -> int:
    mat_root = _mat_root()
    wider_val = loadmat(mat_root / "wider_face_val.mat")
    subset = loadmat(mat_root / f"wider_{level}_val.mat")

    event_list = wider_val["event_list"]
    file_list = wider_val["file_list"]
    face_bbx_list = wider_val["face_bbx_list"]
    gt_list = subset["gt_list"]

    out_root = ROOT / "face-det" / "data" / f"wider_eval_{level}"
    if out_root.exists():
        shutil.rmtree(out_root)
    ensure_split_dirs(out_root)

    count = 0
    for event_idx in range(len(event_list)):
        event_name = _to_text(event_list[event_idx][0])
        files_for_event = file_list[event_idx][0]
        boxes_for_event = face_bbx_list[event_idx][0]
        keep_for_event = gt_list[event_idx][0]

        for file_idx in range(len(files_for_event)):
            file_name = _to_text(files_for_event[file_idx][0])
            image_path = WIDER_ROOT / "WIDER_val" / "images" / event_name / f"{file_name}.jpg"
            if not image_path.exists():
                continue
            raw_boxes = _unwrap_singleton_object_array(boxes_for_event[file_idx])
            keep_raw = _unwrap_singleton_object_array(keep_for_event[file_idx])
            keep_idx = keep_raw.reshape(-1).tolist() if hasattr(keep_raw, "reshape") else []
            keep_idx = [int(v) - 1 for v in keep_idx if int(v) > 0]
            selected = []
            for k in keep_idx:
                if k < 0 or k >= len(raw_boxes):
                    continue
                x, y, w, h = [float(v) for v in raw_boxes[k][:4]]
                if w <= 1 or h <= 1:
                    continue
                selected.append((x, y, w, h))
            if not selected:
                continue

            target_stem = f"{event_name}__{file_name}".replace("/", "__")
            target_image = out_root / "images" / "val" / f"{target_stem}.jpg"
            target_label = out_root / "labels" / "val" / f"{target_stem}.txt"
            shutil.copy2(image_path, target_image)
            with Image.open(image_path) as img:
                width, height = img.size
            write_yolo_label(target_label, selected, width, height)
            count += 1
    return count


def main() -> None:
    for level in ("easy", "hard"):
        count = build_subset(level)
        print(f"{level}: images={count}")


if __name__ == "__main__":
    main()