"""Annotation Detector — detect hand-drawn circles, highlights, and marks on screenshots.

Detects colored annotations (red, blue, green, yellow, pink) drawn on screenshots
and identifies which on-screen elements they surround or point to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Colour presets (HSV ranges for common annotation colours)
# ──────────────────────────────────────────────────────────────────────────────

_COLOR_PRESETS: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    "red": [
        (np.array([0, 60, 40]), np.array([10, 255, 255])),
        (np.array([170, 60, 40]), np.array([180, 255, 255])),
    ],
    "blue": [
        (np.array([100, 60, 40]), np.array([130, 255, 255])),
    ],
    "green": [
        (np.array([40, 60, 40]), np.array([80, 255, 255])),
    ],
    "yellow": [
        (np.array([20, 60, 40]), np.array([35, 255, 255])),
    ],
    "pink": [
        (np.array([145, 40, 60]), np.array([165, 255, 255])),
    ],
    "orange": [
        (np.array([5, 60, 80]), np.array([20, 255, 255])),
    ],
}

# Minimum saturation/value to avoid noise
_MIN_SATURATION = 40
_MIN_VALUE = 30


@dataclass
class Annotation:
    """A single annotation (circle, underline, highlight) found on the image."""
    color: str                 # e.g. "red", "blue"
    shape: str                 # "circle", "ellipse", "underline", "highlight", "freeform"
    bounds: Tuple[int, int, int, int]  # (x, y, width, height) bounding box in image px
    center: Tuple[int, int]    # (cx, cy)
    area: int                  # pixel count of the annotation mark
    confidence: float          # 0.0-1.0 how certain we are this is a deliberate annotation

    def to_dict(self) -> dict:
        return {
            "color": self.color,
            "shape": self.shape,
            "bounds": list(self.bounds),
            "center": list(self.center),
            "area": self.area,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class AnnotationResult:
    """Collection of annotations plus analysis of what they surround."""
    annotations: List[Annotation] = field(default_factory=list)
    annotated_elements: List[dict] = field(default_factory=list)
    available: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "annotations": [a.to_dict() for a in self.annotations],
            "annotated_elements": self.annotated_elements,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_annotations(
    cv_img: np.ndarray,
    ocr_regions: Optional[List[dict]] = None,
    colors: Optional[List[str]] = None,
    min_area: int = 200,
    scale_kb_to_img: Tuple[float, float] = (1.0, 1.0),
) -> AnnotationResult:
    """Detect hand-drawn annotations on a screenshot.

    Args:
        cv_img: BGR OpenCV image array.
        ocr_regions: Optional list of OCR region dicts with keys
            text, x, y, width, height (from OCR engine).
        colors: Which colours to check. Default: ["red", "blue", "green"].
        min_area: Minimum pixel area for a valid annotation mark.
        scale_kb_to_img: (scale_x, scale_y) to convert OCR coords to image coords.

    Returns:
        AnnotationResult with detected annotations and associated elements.
    """
    if not _CV2_AVAILABLE:
        h, w = cv_img.shape[:2] if cv_img is not None else (0, 0)
        return AnnotationResult(
            available=False,
            error="opencv-python not installed. Run: pip install opencv-python",
        )

    if colors is None:
        colors = ["red", "blue", "green"]

    h, w = cv_img.shape[:2]
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    scale_x, scale_y = scale_kb_to_img

    annotations: List[Annotation] = []

    for color_name in colors:
        presets = _COLOR_PRESETS.get(color_name)
        if not presets:
            continue

        # Build combined mask for this colour
        mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in presets:
            mask |= cv2.inRange(hsv, lower, upper)

        # Clean up: remove single-pixel noise, connect nearby strokes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))

        # Find contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # Compute circularity: 1.0 = perfect circle
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Classify shape
            shape = _classify_shape(circularity, bw, bh, area, perimeter)

            # Compute confidence: how annotation-like is this?
            confidence = _annotation_confidence(circularity, shape, area, bw, bh, h, w)

            cx, cy = x + bw // 2, y + bh // 2

            annotations.append(Annotation(
                color=color_name,
                shape=shape,
                bounds=(x, y, bw, bh),
                center=(cx, cy),
                area=int(area),
                confidence=confidence,
            ))

    # Sort by confidence (most likely annotations first)
    annotations.sort(key=lambda a: a.confidence, reverse=True)

    # Cross-reference with OCR to find which elements are annotated
    annotated_elements = []
    if ocr_regions and annotations:
        for ann in annotations:
            elements_inside = _find_ocr_inside(
                ann.bounds, ann.center, ocr_regions, scale_x, scale_y
            )
            if elements_inside:
                annotated_elements.append({
                    "annotation_color": ann.color,
                    "annotation_shape": ann.shape,
                    "confidence": round(ann.confidence, 2),
                    "elements": elements_inside,
                })
            else:
                # Check for elements near the annotation (pointing/underlining)
                elements_near = _find_ocr_near(
                    ann.bounds, ann.center, ocr_regions, scale_x, scale_y
                )
                if elements_near:
                    annotated_elements.append({
                        "annotation_color": ann.color,
                        "annotation_shape": ann.shape,
                        "confidence": round(ann.confidence, 2),
                        "elements": elements_near,
                        "proximity": "nearby",
                    })

    return AnnotationResult(
        annotations=annotations,
        annotated_elements=annotated_elements,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shape classification
# ──────────────────────────────────────────────────────────────────────────────

def _classify_shape(
    circularity: float, bw: int, bh: int,
    area: float, perimeter: float,
) -> str:
    """Classify the shape of a contour."""
    aspect = bw / max(bh, 1)

    # Ring-shaped (hand-drawn circle): high circularity, thin stroke
    # A ring has a large perimeter relative to its area
    thinness = perimeter * perimeter / max(area, 1)

    if circularity > 0.6 and thinness > 25 and thinness < 200:
        # Could be a circle or ellipse
        if 0.7 < aspect < 1.4:
            return "circle"
        else:
            return "ellipse"
    elif circularity > 0.4 and area < 5000:
        return "circle"
    elif bw < 50 and bh < 30:
        return "dot"
    elif bh < bw * 0.4 and area < 5000:
        return "underline"
    elif bw > 100 and bh > 50 and area > 5000:
        return "highlight"
    else:
        return "freeform"


def _annotation_confidence(
    circularity: float, shape: str, area: float,
    bw: int, bh: int, img_w: int, img_h: int,
) -> float:
    """Score how likely this is a deliberate annotation (0.0-1.0).

    Hand-drawn annotations tend to:
    - Be relatively small compared to the image (not UI elements)
    - Have moderate circularity (not perfectly circular like UI buttons)
    - NOT be too large (whole-screen red isn't an annotation)
    """
    score = 0.5  # start neutral

    # Shape bonus
    if shape in ("circle", "ellipse"):
        score += 0.2
    elif shape == "underline":
        score += 0.15
    elif shape == "highlight":
        score += 0.1

    # Circularity: hand-drawn circles are ~0.3-0.7 (not perfect like UI)
    if 0.25 < circularity < 0.75:
        score += 0.1
    elif circularity >= 0.85:
        score -= 0.1  # too perfect — likely a UI element

    # Size penalty: too large = probably not an annotation
    img_area = img_w * img_h
    area_ratio = area / img_area if img_area > 0 else 0
    if area_ratio > 0.3:
        score -= 0.4  # massive region = layout, not annotation
    elif area_ratio > 0.1:
        score -= 0.2

    # Too small = probably noise
    if area < 300:
        score -= 0.2

    return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────────────────────────────────────
# OCR cross-referencing
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_to_img_coords(
    region: dict, scale_x: float, scale_y: float,
) -> Tuple[int, int, int, int]:
    """Convert OCR region coordinates to image coordinates."""
    x = int(region.get("x", 0) * scale_x)
    y = int(region.get("y", 0) * scale_y)
    w = int(region.get("width", 0) * scale_x)
    h = int(region.get("height", 0) * scale_y)
    return x, y, x + w, y + h


def _find_ocr_inside(
    bounds: Tuple[int, int, int, int],
    center: Tuple[int, int],
    ocr_regions: List[dict],
    scale_x: float,
    scale_y: float,
) -> List[dict]:
    """Find OCR text elements whose bounding box falls inside the annotation."""
    bx, by, bw, bh = bounds
    cx, cy = center
    result = []

    for r in ocr_regions:
        rx1, ry1, rx2, ry2 = _ocr_to_img_coords(r, scale_x, scale_y)
        # Check if the OCR center is within the annotation bounds
        ocr_cx = (rx1 + rx2) // 2
        ocr_cy = (ry1 + ry2) // 2

        if bx <= ocr_cx <= bx + bw and by <= ocr_cy <= by + bh:
            result.append({
                "text": r.get("text", ""),
                "confidence": r.get("confidence", 0),
                "position": [rx1, ry1, rx2 - rx1, ry2 - ry1],
            })

    return result


def _find_ocr_near(
    bounds: Tuple[int, int, int, int],
    center: Tuple[int, int],
    ocr_regions: List[dict],
    scale_x: float,
    scale_y: float,
    margin: int = 50,
) -> List[dict]:
    """Find OCR text elements near the annotation (within margin pixels)."""
    bx, by, bw, bh = bounds
    cx, cy = center
    result = []

    for r in ocr_regions:
        rx1, ry1, rx2, ry2 = _ocr_to_img_coords(r, scale_x, scale_y)
        ocr_cx = (rx1 + rx2) // 2
        ocr_cy = (ry1 + ry2) // 2

        # Check if close to the annotation
        dx = abs(ocr_cx - cx)
        dy = abs(ocr_cy - cy)
        if dx < max(bw, 100) + margin and dy < max(bh, 100) + margin:
            if not (bx <= ocr_cx <= bx + bw and by <= ocr_cy <= by + bh):
                result.append({
                    "text": r.get("text", ""),
                    "confidence": r.get("confidence", 0),
                    "position": [rx1, ry1, rx2 - rx1, ry2 - ry1],
                })

    return result
