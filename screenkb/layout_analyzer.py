"""Layout Analyzer — detect UI panels, menus, toolbars, sidebars via OpenCV."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


@dataclass
class Region:
    label: str          # e.g. "sidebar", "toolbar", "content", "panel"
    x: int
    y: int
    width: int
    height: int
    area_ratio: float   # fraction of total image area

    def to_dict(self):
        return {
            "type": self.label,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "area_ratio": round(self.area_ratio, 3),
        }


@dataclass
class LayoutResult:
    image_width: int
    image_height: int
    regions: List[Region] = field(default_factory=list)
    region_labels: List[str] = field(default_factory=list)
    available: bool = True
    error: Optional[str] = None

    def to_dict(self):
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "region_count": len(self.regions),
            "region_labels": sorted(set(self.region_labels)),
            "regions": [r.to_dict() for r in self.regions],
        }


def analyze_layout(cv_img: np.ndarray) -> LayoutResult:
    """Detect layout regions in a BGR OpenCV image.

    Strategy:
    1. Convert to grayscale, apply edge detection
    2. Find large rectangular contours
    3. Classify by position/proportions: sidebar, toolbar, content, statusbar, menubar
    """
    if not _CV2_AVAILABLE:
        h, w = cv_img.shape[:2] if cv_img is not None else (0, 0)
        return LayoutResult(
            image_width=w,
            image_height=h,
            available=False,
            error="opencv-python not installed. Run: pip install opencv-python",
        )

    h, w = cv_img.shape[:2]
    total_area = w * h

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Detect structural lines — not fine edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    # Dilate to connect nearby edges into regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        ratio = area / total_area

        # Ignore tiny regions (< 2% of image)
        if ratio < 0.02:
            continue

        label = _classify_region(x, y, rw, rh, w, h)
        regions.append(Region(label=label, x=x, y=y, width=rw, height=rh, area_ratio=ratio))

    # Deduplicate overlapping regions — keep largest when overlap > 80%
    regions = _deduplicate(regions)
    region_labels = [r.label for r in regions]

    # If no structural regions detected, at minimum note the full canvas
    if not regions:
        regions.append(Region(label="content", x=0, y=0, width=w, height=h, area_ratio=1.0))
        region_labels = ["content"]

    return LayoutResult(
        image_width=w,
        image_height=h,
        regions=regions,
        region_labels=region_labels,
    )


def _classify_region(x: int, y: int, rw: int, rh: int, img_w: int, img_h: int) -> str:
    """Classify a bounding box by position and proportions."""
    rel_x = x / img_w
    rel_y = y / img_h
    rel_w = rw / img_w
    rel_h = rh / img_h

    # Toolbar / menubar: top strip, full width, short height
    if rel_y < 0.08 and rel_w > 0.7 and rel_h < 0.12:
        return "toolbar" if rel_y > 0.02 else "menubar"

    # Status bar: bottom strip
    if rel_y > 0.90 and rel_w > 0.5 and rel_h < 0.06:
        return "statusbar"

    # Sidebar: left or right strip, tall, and narrow (< 30% width)
    # Reject if area_ratio > 0.4 — a region covering 40%+ of screen is not a sidebar
    area_ratio = (rw * rh) / (img_w * img_h)
    if area_ratio < 0.4:
        if rel_x < 0.12 and rel_h > 0.5 and rel_w < 0.3:
            return "sidebar"
        if rel_x > 0.85 and rel_h > 0.5 and rel_w < 0.15:
            return "sidebar"
    # Very narrow tall strip (scrollbar / edge artifact) → panel
    if rw < 40 and rel_h > 0.5:
        return "panel"

    # Panel: narrower section (could be file explorer, output pane, etc.)
    if rel_w < 0.35 and rel_h > 0.3:
        return "panel"

    # Wide short region at bottom — terminal / console
    if rel_y > 0.55 and rel_w > 0.5 and rel_h < 0.45:
        return "terminal"

    return "content"


def _deduplicate(regions: List[Region]) -> List[Region]:
    """Remove regions that are heavily contained within another."""
    kept = []
    sorted_regions = sorted(regions, key=lambda r: r.area_ratio, reverse=True)
    for candidate in sorted_regions:
        dominated = False
        for existing in kept:
            if _overlap_ratio(candidate, existing) > 0.8:
                dominated = True
                break
        if not dominated:
            kept.append(candidate)
    return kept


def _overlap_ratio(a: Region, b: Region) -> float:
    """Intersection / area_a."""
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.width, b.x + b.width)
    iy2 = min(a.y + a.height, b.y + b.height)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = a.width * a.height
    return inter / area_a if area_a > 0 else 0.0
