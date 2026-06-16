"""Demo: 106-point face landmark detection.

Usage examples:
    # Single image
    py -3.12 scripts/demo.py --image path/to/face.jpg --output result.jpg

    # Webcam (real-time)
    py -3.12 scripts/demo.py --webcam

    # ONNX INT8 inference
    py -3.12 scripts/demo.py --image path/to/face.jpg --onnx runs/lapa_hrnet_w18_awing_mixed_e80/model_int8.onnx
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def crop_resize_face(image: Image.Image, bbox: tuple[float, float, float, float],
                     image_size: int = 256, crop_scale: float = 1.35) -> tuple[Image.Image, tuple]:
    """Crop face region using bbox + crop_scale, then resize to image_size."""
    from torchvision.transforms import functional as TF
    from torchvision.transforms import InterpolationMode

    w, h = image.size
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * crop_scale

    left = max(0, cx - side / 2)
    top = max(0, cy - side / 2)
    right = min(w, left + side)
    bottom = min(h, top + side)
    crop_w = right - left
    crop_h = bottom - top

    cropped = TF.resized_crop(
        image,
        top=int(round(top)),
        left=int(round(left)),
        height=max(2, int(round(crop_h))),
        width=max(2, int(round(crop_w))),
        size=[image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return cropped, (left, top, crop_w, crop_h)


def detect_face_with_haar(image: Image.Image) -> tuple[float, float, float, float] | None:
    """Use OpenCV Haar cascade to detect a face. Returns (x1, y1, x2, y2) in pixel coords."""
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed; using full image as bbox")
        w, h = image.size
        return (0, 0, w, h)

    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    # Pick the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    return (float(x), float(y), float(x + w), float(y + h))


def predict_pytorch(model, image_tensor: "torch.Tensor", device) -> np.ndarray:
    """Run PyTorch model and return (106, 2) landmark predictions in [0, 1]."""
    import torch
    with torch.no_grad():
        out = model(image_tensor.unsqueeze(0).to(device))
    return out.cpu().numpy()[0]  # (106, 2)


def predict_onnx(session, image_tensor_np: np.ndarray) -> np.ndarray:
    """Run ONNX session and return (106, 2) landmark predictions in [0, 1]."""
    out = session.run(None, {"images": image_tensor_np[None].astype(np.float32)})
    return out[0][0]  # (106, 2)


def draw_landmarks(image: Image.Image, landmarks_px: np.ndarray, point_radius: int = 2,
                   color_outline: str = "lime", color_eye: str = "red") -> Image.Image:
    """Draw 106 landmarks on the image. Use color_eye for eye points (66~93)."""
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for i, (x, y) in enumerate(landmarks_px):
        color = color_eye if 66 <= i <= 93 else color_outline
        r = point_radius
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)
    return image


def run_single(image_path: Path, args) -> None:
    image = Image.open(image_path).convert("RGB")
    bbox = detect_face_with_haar(image)
    if bbox is None:
        print(f"No face detected in {image_path}")
        return

    cropped, (cx, cy, cw, ch) = crop_resize_face(image, bbox, args.image_size, args.crop_scale)

    if args.onnx:
        import onnxruntime as ort
        from torchvision.transforms import functional as TF
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        tensor = TF.to_tensor(cropped)
        tensor = TF.normalize(tensor, mean=[0.5] * 3, std=[0.5] * 3).numpy()
        t0 = time.time()
        landmarks = predict_onnx(sess, tensor)
        latency = (time.time() - t0) * 1000
        print(f"ONNX inference: {latency:.1f} ms")
    else:
        import torch
        from torchvision.transforms import functional as TF
        from landmarklab.core import load_config
        from landmarklab.model import build_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = build_model(ckpt["config"]).to(device).eval()
        model.load_state_dict(ckpt["model"])
        tensor = TF.to_tensor(cropped)
        tensor = TF.normalize(tensor, mean=[0.5] * 3, std=[0.5] * 3)
        t0 = time.time()
        landmarks = predict_pytorch(model, tensor, device)
        latency = (time.time() - t0) * 1000
        print(f"PyTorch inference ({device}): {latency:.1f} ms")

    # Convert normalized coords back to original image pixel coords
    landmarks_px = landmarks.copy()
    landmarks_px[:, 0] = landmarks[:, 0] * cw + cx
    landmarks_px[:, 1] = landmarks[:, 1] * ch + cy

    # Draw and save
    out_image = draw_landmarks(image, landmarks_px, point_radius=args.point_radius)
    out_path = Path(args.output)
    out_image.save(out_path)
    print(f"Saved: {out_path} ({len(landmarks_px)} landmarks)")


def run_webcam(args) -> None:
    try:
        import cv2
    except ImportError:
        print("OpenCV required for webcam mode: pip install opencv-python")
        return
    import torch
    from torchvision.transforms import functional as TF
    from landmarklab.core import load_config
    from landmarklab.model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt["model"])

    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cropped, (cx, cy, cw, ch) = crop_resize_face(
                image, (x, y, x + w, y + h), args.image_size, args.crop_scale
            )
            tensor = TF.to_tensor(cropped)
            tensor = TF.normalize(tensor, mean=[0.5] * 3, std=[0.5] * 3)
            with torch.no_grad():
                out = model(tensor.unsqueeze(0).to(device))
            landmarks = out.cpu().numpy()[0]
            for i, (lx, ly) in enumerate(landmarks):
                px = int(lx * cw + cx)
                py = int(ly * ch + cy)
                color = (0, 0, 255) if 66 <= i <= 93 else (0, 255, 0)
                cv2.circle(frame, (px, py), 2, color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("face106 demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="face106 landmark detection demo")
    parser.add_argument("--image", help="Input image path")
    parser.add_argument("--output", default="demo_output.jpg", help="Output image path")
    parser.add_argument("--webcam", action="store_true", help="Use webcam (requires OpenCV)")
    parser.add_argument("--checkpoint", default="runs/lapa_hrnet_w18_awing_mixed_e80/best.pt",
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--onnx", help="Path to ONNX INT8 model (overrides --checkpoint)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--point-radius", type=int, default=2)
    args = parser.parse_args()

    if args.webcam:
        run_webcam(args)
    elif args.image:
        run_single(Path(args.image), args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
