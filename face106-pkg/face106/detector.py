"""LandmarkDetector: high-level API for 106-point face landmark detection."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

DEFAULT_MODEL_NAME = "face106_int8.onnx"
PACKAGE_ASSET_DIR = Path(__file__).parent / "assets"


class LandmarkDetector:
    """106-point facial landmark detector with INT8 ONNX inference.

    Args:
        model_path: Path to ONNX model. If None, looks up bundled INT8 model.
        image_size: Input image size (default 256, matches training).
        crop_scale: Bbox expansion factor for face crop (default 1.35).
        provider: ONNXRuntime provider, e.g. "CPUExecutionProvider" or "CUDAExecutionProvider".

    Usage:
        >>> from face106 import LandmarkDetector
        >>> det = LandmarkDetector()
        >>> from PIL import Image
        >>> img = Image.open("face.jpg").convert("RGB")
        >>> landmarks = det.predict(img, bbox=(50, 50, 250, 250))
        >>> landmarks.shape
        (106, 2)
    """

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        image_size: int = 256,
        crop_scale: float = 1.35,
        provider: str = "CPUExecutionProvider",
    ):
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.model_path = self._resolve_model_path(model_path)

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime") from exc

        providers = [provider] if provider else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self._input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _resolve_model_path(model_path) -> Path:
        if model_path is not None:
            p = Path(model_path)
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")
            return p
        # Try bundled asset
        bundled = PACKAGE_ASSET_DIR / DEFAULT_MODEL_NAME
        if bundled.exists():
            return bundled
        raise FileNotFoundError(
            f"No model_path given and bundled asset not found at {bundled}. "
            f"Place the INT8 ONNX model there or pass model_path explicitly."
        )

    @staticmethod
    def _crop_resize(image: Image.Image, bbox: Tuple[float, float, float, float],
                     image_size: int, crop_scale: float):
        """Crop face region using bbox + crop_scale, then resize to image_size."""
        w, h = image.size
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * crop_scale

        left = max(0.0, cx - side / 2)
        top = max(0.0, cy - side / 2)
        right = min(float(w), left + side)
        bottom = min(float(h), top + side)
        crop_w = max(2.0, right - left)
        crop_h = max(2.0, bottom - top)

        # Use PIL crop + resize (no torchvision dependency)
        cropped = image.crop((int(round(left)), int(round(top)),
                              int(round(left + crop_w)), int(round(top + crop_h))))
        cropped = cropped.resize((image_size, image_size), Image.BILINEAR)
        return cropped, (left, top, crop_w, crop_h)

    @staticmethod
    def _to_normalized_tensor(image: Image.Image) -> np.ndarray:
        """Convert PIL image to (1, 3, H, W) float32 normalized to [-1, 1]."""
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0  # (H, W, 3)
        arr = (arr - 0.5) / 0.5
        arr = arr.transpose(2, 0, 1)  # (3, H, W)
        return arr[None].astype(np.float32)

    def predict(self, image: Image.Image,
                bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Predict 106 landmarks for a face in the image.

        Args:
            image: PIL.Image (RGB).
            bbox: face bounding box (x1, y1, x2, y2) in pixel coords.

        Returns:
            np.ndarray of shape (106, 2) — landmark pixel coordinates in the original image.
        """
        cropped, (cx, cy, cw, ch) = self._crop_resize(
            image, bbox, self.image_size, self.crop_scale
        )
        tensor = self._to_normalized_tensor(cropped)
        out = self.session.run(None, {self._input_name: tensor})[0]  # (1, 106, 2)
        landmarks = out[0]  # (106, 2) normalized [0, 1]

        # Convert back to original image pixel coords
        result = np.empty_like(landmarks)
        result[:, 0] = landmarks[:, 0] * cw + cx
        result[:, 1] = landmarks[:, 1] * ch + cy
        return result

    def predict_batch(self, images_and_bboxes) -> list:
        """Predict landmarks for a list of (image, bbox) tuples."""
        return [self.predict(img, bb) for img, bb in images_and_bboxes]


def draw_landmarks(image: Image.Image, landmarks: np.ndarray,
                   point_radius: int = 2,
                   color: str = "lime",
                   eye_color: str = "red") -> Image.Image:
    """Visualize 106 landmarks on the image. Eye points (66~93) get eye_color."""
    from PIL import ImageDraw
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for i, (x, y) in enumerate(landmarks):
        c = eye_color if 66 <= i <= 93 else color
        r = point_radius
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c, outline=c)
    return out
