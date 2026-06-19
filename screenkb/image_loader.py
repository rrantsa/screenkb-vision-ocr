"""Image Loader — load, normalize, and resize screenshots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_WIDTH = 1920
MAX_HEIGHT = 1080


class ImageLoadError(Exception):
    pass


def load_image(path: str | Path) -> Tuple[Image.Image, np.ndarray]:
    """Load an image file, normalize to RGB, resize if needed.

    Returns:
        (pil_image, cv_image) — PIL for OCR, NumPy array (BGR) for OpenCV.
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    try:
        pil_img = Image.open(path).convert("RGB")
    except Exception as exc:
        raise ImageLoadError(f"Cannot open image: {exc}") from exc

    # Resize if larger than target — preserve aspect ratio
    pil_img = _maybe_resize(pil_img, MAX_WIDTH, MAX_HEIGHT)

    # Convert to OpenCV-compatible NumPy array (BGR)
    cv_img = np.array(pil_img)[:, :, ::-1].copy()  # RGB → BGR

    return pil_img, cv_img


def _maybe_resize(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Downscale if necessary, preserving aspect ratio."""
    w, h = img.size
    if w <= max_w and h <= max_h:
        return img
    ratio = min(max_w / w, max_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    return img.resize((new_w, new_h), Image.LANCZOS)
