"""CV Analyzer — OpenCV computer vision for general photo/scene understanding.

Detects faces, skin regions, edges, MSER text candidates, and classifies
images that are NOT UI screenshots (photos of people, documents, landscapes, etc.).

This extends screenkb beyond its original UI-screenshot focus.
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


def _to_native(val):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceInfo:
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return {
            "x": _to_native(self.x),
            "y": _to_native(self.y),
            "width": _to_native(self.width),
            "height": _to_native(self.height),
        }


@dataclass
class CVResult:
    """Result of computer vision analysis on a general image (non-UI)."""

    is_photo: bool                     # True if this looks like a photo, not a UI screenshot
    scene_type: str                    # "people_photo" | "document" | "landscape" | "object" | "ui_screenshot" | "unknown"
    scene_confidence: float            # 0.0-1.0

    faces: List[FaceInfo] = field(default_factory=list)
    face_count: int = 0

    skin_ratio: float = 0.0            # % of image covered by skin-tone pixels
    edge_intensity: float = 0.0        # 0-255 average of Canny edge image
    edge_ratio: float = 0.0            # % of pixels that are edges
    contour_count: int = 0             # number of detected contours (after filtering)
    largest_contours: List[dict] = field(default_factory=list)  # top 3

    mser_count: int = 0                # MSER region count (text-like candidates)
    brightness: float = 0.0            # 0-255 average brightness
    dominant_color: Tuple[int, ...] = (0, 0, 0)  # BGR average

    features: List[str] = field(default_factory=list)  # generated feature names

    available: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_photo": self.is_photo,
            "scene_type": self.scene_type,
            "scene_confidence": round(self.scene_confidence, 2),
            "face_count": self.face_count,
            "faces": [f.to_dict() for f in self.faces],
            "skin_ratio": round(_to_native(self.skin_ratio), 1),
            "edge_intensity": round(_to_native(self.edge_intensity), 1),
            "edge_ratio": round(_to_native(self.edge_ratio), 1),
            "contour_count": self.contour_count,
            "largest_contours": self.largest_contours[:3],
            "mser_count": self.mser_count,
            "brightness": int(round(_to_native(self.brightness))),
            "dominant_color": [int(_to_native(v)) for v in self.dominant_color],
            "features": self.features,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def analyze_cv(cv_img: np.ndarray) -> CVResult:
    """Run full computer vision analysis on a BGR OpenCV image.

    Detects faces, skin, edges, MSER, and classifies the image type.
    Returns a CVResult with scene classification and raw data.
    """
    if not _CV2_AVAILABLE:
        return CVResult(
            is_photo=False,
            scene_type="unknown",
            scene_confidence=0.0,
            available=False,
            error="opencv-python not installed. Run: pip install opencv-python",
        )

    h, w = cv_img.shape[:2]
    if h == 0 or w == 0:
        return CVResult(
            is_photo=False, scene_type="unknown", scene_confidence=0.0,
            available=False, error="Empty image",
        )

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    total_px = w * h

    # ── Face detection (Haar cascades) ──
    faces = _detect_faces(gray)
    face_count = len(faces)
    max_face_size_ratio = 0.0
    if faces:
        areas = [fw * fh for (_, _, fw, fh) in faces]
        max_area = max(areas)
        max_face_size_ratio = max_area / total_px if total_px > 0 else 0

    # ── Skin detection ──
    skin_ratio = _detect_skin(hsv, total_px)

    # ── Edge analysis (Canny) ──
    edge_intensity, edge_ratio = _edge_analysis(gray, total_px)

    # ── Contour analysis ──
    contour_count, largest_contours = _contour_analysis(gray)

    # ── MSER (text-like regions) ──
    mser_count = _mser_analysis(gray)

    # ── Color / brightness ──
    avg_brightness = _to_native(float(np.mean(gray)))
    mean_bgr = tuple(_to_native(v) for v in cv2.mean(cv_img)[:3])

    # ── Scene classification ──
    is_photo, scene_type, scene_confidence = _classify_scene(
        face_count=face_count,
        max_face_size_ratio=max_face_size_ratio,
        skin_ratio=skin_ratio,
        edge_ratio=edge_ratio,
        edge_intensity=edge_intensity,
        contour_count=contour_count,
        mser_count=mser_count,
        avg_brightness=avg_brightness,
        total_px=total_px,
        w=w, h=h,
    )

    # ── Generate feature names ──
    features = _generate_cv_features(
        is_photo, scene_type, face_count, skin_ratio,
        contour_count, edge_ratio, mser_count, avg_brightness,
    )

    return CVResult(
        is_photo=is_photo,
        scene_type=scene_type,
        scene_confidence=scene_confidence,
        faces=[FaceInfo(x=fx, y=fy, width=fw, height=fh) for (fx, fy, fw, fh) in faces],
        face_count=face_count,
        skin_ratio=skin_ratio,
        edge_intensity=edge_intensity,
        edge_ratio=edge_ratio,
        contour_count=contour_count,
        largest_contours=largest_contours[:3],
        mser_count=mser_count,
        brightness=avg_brightness,
        dominant_color=tuple(int(v) for v in mean_bgr),
        features=features,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Face detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_faces(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces using Haar cascades. Returns list of (x, y, w, h)."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        return []
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(x, y, w, h) for (x, y, w, h) in faces]


# ──────────────────────────────────────────────────────────────────────────────
# Skin detection
# ──────────────────────────────────────────────────────────────────────────────

# Typical skin HSV ranges
_SKIN_RANGES: List[Tuple[np.ndarray, np.ndarray]] = [
    (np.array([0, 20, 70], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)),
    (np.array([170, 20, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
]


def _detect_skin(hsv: np.ndarray, total_px: int) -> float:
    """Detect skin-tone pixels and return ratio (0.0-100.0)."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in _SKIN_RANGES:
        mask |= cv2.inRange(hsv, lower, upper)
    skin_count = np.count_nonzero(mask)
    return (skin_count / total_px * 100.0) if total_px > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Edge analysis
# ──────────────────────────────────────────────────────────────────────────────

def _edge_analysis(gray: np.ndarray, total_px: int) -> Tuple[float, float]:
    """Run Canny edge detection and return (mean_intensity, edge_pixel_ratio%)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_count = np.count_nonzero(edges)
    edge_ratio = (edge_count / total_px * 100.0) if total_px > 0 else 0.0
    edge_intensity = float(np.mean(edges))
    return _to_native(edge_intensity), _to_native(edge_ratio)


# ──────────────────────────────────────────────────────────────────────────────
# Contour analysis
# ──────────────────────────────────────────────────────────────────────────────

def _contour_analysis(gray: np.ndarray) -> Tuple[int, List[dict]]:
    """Find contours and return (count, top_5_contours_info)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter tiny noise contours
    h, w = gray.shape
    total_area = w * h
    min_area = total_area * 0.001  # 0.1% of image
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Top 5 by area
    sorted_c = sorted(filtered, key=cv2.contourArea, reverse=True)
    top_info = []
    for c in sorted_c[:5]:
        area = cv2.contourArea(c)
        x, y, cw, ch = cv2.boundingRect(c)
        top_info.append({
            "area": _to_native(int(area)),
            "area_ratio": _to_native(round(area / total_area, 3)) if total_area > 0 else 0,
            "bounds": [_to_native(x), _to_native(y), _to_native(cw), _to_native(ch)],
        })

    return len(filtered), top_info


# ──────────────────────────────────────────────────────────────────────────────
# MSER text candidate detection
# ──────────────────────────────────────────────────────────────────────────────

def _mser_analysis(gray: np.ndarray) -> int:
    """Detect MSER regions (text-like candidates) and return count."""
    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        # Filter: typical text region sizes
        text_candidates = [
            r for r in regions
            if 50 < cv2.contourArea(r) < 5000
        ]
        return len(text_candidates)
    except Exception:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Scene classification
# ──────────────────────────────────────────────────────────────────────────────

def _classify_scene(
    face_count: int,
    max_face_size_ratio: float,
    skin_ratio: float,
    edge_ratio: float,
    edge_intensity: float,
    contour_count: int,
    mser_count: int,
    avg_brightness: float,
    total_px: int,
    w: int, h: int,
) -> Tuple[bool, str, float]:
    """Classify the image type: people_photo, document, landscape, etc.

    Returns (is_photo, scene_type, confidence).
    """
    # ── High-confidence photo of people ──
    if face_count >= 1 and skin_ratio > 5:
        return True, "people_photo", 0.85
    if face_count >= 2:
        return True, "people_photo", 0.90

    # ── Photo with skin but no face detected ──
    # Could be people (from behind/side) OR warm-toned building/landscape
    if skin_ratio > 15 and edge_ratio > 0.5:
        if face_count == 0 and mser_count > 50 and contour_count < 10:
            # High MSER + few contours + no faces + warm tones → building/architecture
            return True, "building_photo", 0.60
        return True, "people_photo", 0.65

    # ── Document / paper photo ──
    # High brightness, many MSER candidates (text-like), moderate edges
    if avg_brightness > 180 and mser_count > 30 and contour_count > 30:
        return True, "document", 0.70

    # ── Screenshot with lots of text ──
    # Many MSER regions, moderate edges, lots of contours, any brightness
    if mser_count > 50 and contour_count > 50 and total_px > 100000:
        return False, "ui_screenshot", 0.60  # lots of text-like regions

    # ── Outdoor / landscape ──
    # High brightness, moderate to many edges, few MSER
    if avg_brightness > 180 and edge_ratio > 1.0 and mser_count < 20:
        return True, "landscape", 0.55

    # ── Screenshot with minimal text (app UI with icons) ──
    # Few MSER, moderate edges, moderate contours
    if 5 < edge_ratio < 20 and mser_count < 15 and contour_count < 40:
        return False, "ui_screenshot", 0.50

    # ── Very smooth image (blurry, flat color, or professional photo) ──
    if edge_ratio < 0.5 and avg_brightness > 200:
        return True, "landscape", 0.40  # likely out-of-focus background / sky

    # ── Dark image ──
    if avg_brightness < 80:
        return True, "object", 0.45

    # ── Generic fallback: compare contour density ──
    # UI screenshots tend to have structured rectangular regions
    # Photos tend to have organic, irregular contours
    if contour_count > 80:
        return True, "landscape", 0.50  # many small details = outdoor scene
    if contour_count < 10 and mser_count < 5:
        return True, "object", 0.40  # very simple = close-up of object

    # Default: UI screenshot (less confident)
    return False, "ui_screenshot", 0.40


# ──────────────────────────────────────────────────────────────────────────────
# Feature generation (for reasoning engine integration)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_cv_features(
    is_photo: bool,
    scene_type: str,
    face_count: int,
    skin_ratio: float,
    contour_count: int,
    edge_ratio: float,
    mser_count: int,
    avg_brightness: float,
) -> List[str]:
    """Generate feature names from CV analysis for the reasoning engine."""
    features = []

    if is_photo:
        features.append("photo")
    else:
        features.append("ui_screenshot")

    features.append(f"scene_{scene_type}")

    if face_count > 0:
        features.append("face_detected")
    if face_count >= 2:
        features.append("multiple_faces")
    if skin_ratio > 10:
        features.append("skin_present")
    if skin_ratio > 20:
        features.append("skin_abundant")
    if edge_ratio < 0.5:
        features.append("low_edges")
    elif edge_ratio < 2.0:
        features.append("moderate_edges")
    else:
        features.append("high_edges")

    if contour_count > 50:
        features.append("many_contours")
    elif contour_count > 20:
        features.append("moderate_contours")

    if mser_count > 30:
        features.append("many_mser_regions")
    elif mser_count > 10:
        features.append("some_mser_regions")

    return features
