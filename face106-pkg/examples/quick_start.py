"""Quick example for face106 package.

Run:
    pip install -e .
    python examples/quick_start.py path/to/face.jpg
"""
import sys
from pathlib import Path

from PIL import Image

from face106 import LandmarkDetector
from face106.detector import draw_landmarks


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_start.py <image_path> [output_path]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "result.jpg")

    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image.size}")

    # For demo purposes, use the entire image as bbox.
    # In production, use a face detector (RetinaFace, MTCNN, BlazeFace, etc.).
    w, h = image.size
    bbox = (w * 0.1, h * 0.1, w * 0.9, h * 0.9)

    detector = LandmarkDetector()
    landmarks = detector.predict(image, bbox)
    print(f"Detected {len(landmarks)} landmarks")
    print(f"First 3 points: {landmarks[:3]}")

    visualized = draw_landmarks(image, landmarks)
    visualized.save(output_path)
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
