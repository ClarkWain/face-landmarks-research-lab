"""Visualize LaPa 106 landmarks on one image, drawing each region in a different color and
labeling key index landmarks so we can identify the eye / mouth groups for NME."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parents[1]))

from landmarklab.data import _load_lapa_landmarks


def main() -> None:
    sample_landmark = Path("data/LaPa/test/landmarks/13188486554_6.txt")
    sample_image = Path("data/LaPa/test/images/13188486554_6.jpg")

    landmarks = _load_lapa_landmarks(sample_landmark)
    print(f"loaded {landmarks.shape[0]} landmarks")

    image = Image.open(sample_image).convert("RGB")
    print(f"image size: {image.size}")
    draw = ImageDraw.Draw(image)

    # Assumed LaPa 106 index layout (per LaPa GitHub README):
    # 0..32: face contour (33 pts)
    # 33..41: right brow (9 pts)
    # 42..50: left brow (9 pts)
    # 51..65: nose (15 pts)
    # 66..79: right eye (14 pts)
    # 80..93: left eye (14 pts)
    # 94..103: outer mouth (10 pts) + 104..105 inner? — verify visually
    groups = [
        (range(0, 33), (255, 0, 0)),     # contour - red
        (range(33, 42), (255, 165, 0)),  # right brow - orange
        (range(42, 51), (255, 215, 0)),  # left brow - gold
        (range(51, 66), (60, 179, 113)), # nose - sea green
        (range(66, 80), (30, 144, 255)), # right eye - dodger blue
        (range(80, 94), (138, 43, 226)), # left eye - blue violet
        (range(94, 106), (220, 20, 60)), # mouth - crimson
    ]

    for indices, color in groups:
        for i in indices:
            x, y = landmarks[i].tolist()
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=(255, 255, 255))

    # Label a few key candidate indices for eye centers (just to inspect)
    for i in [66, 70, 73, 79, 80, 86, 90, 93]:
        x, y = landmarks[i].tolist()
        draw.text((x + 4, y), str(i), fill=(0, 0, 0))

    out = Path("face106/docs/images/lapa_sample.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
