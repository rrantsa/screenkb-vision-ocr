"""OCR Engine — extract text regions via Tesseract."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

try:
    import pytesseract
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    _PYTESSERACT_AVAILABLE = False

# Allow override via env var (useful on Windows where tesseract is not on PATH)
_TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "")
_COMMON_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def _find_tesseract() -> Optional[str]:
    """Try to locate tesseract on Windows if not on PATH."""
    for p in _COMMON_WINDOWS_PATHS:
        if os.path.exists(p):
            return p
    return None


def _configure_tesseract() -> bool:
    """Configure pytesseract path. Returns True if tesseract is usable."""
    if not _PYTESSERACT_AVAILABLE:
        return False
    if _TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
        return True
    found = _find_tesseract()
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        return True
    return True  # trust PATH — will fail at runtime with a clear error


_TESSERACT_CONFIGURED = _configure_tesseract()


@dataclass
class TextRegion:
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float  # 0–100 from Tesseract

    @property
    def center(self):
        return (self.x + self.width // 2, self.y + self.height // 2)

    def to_dict(self):
        return {
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": round(self.confidence, 1),
        }


@dataclass
class OCRResult:
    raw_text: str
    regions: List[TextRegion] = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    available: bool = True
    error: Optional[str] = None

    def to_dict(self):
        return {
            "raw_text": self.raw_text,
            "word_count": len(self.words),
            "region_count": len(self.regions),
            "regions": [r.to_dict() for r in self.regions],
        }


def run_ocr(pil_img: Image.Image) -> OCRResult:
    """Run Tesseract OCR on a PIL image.

    Falls back to an empty result with error message if Tesseract is not
    available — the rest of the pipeline still works (layout-only analysis).
    """
    if not _PYTESSERACT_AVAILABLE:
        return OCRResult(
            raw_text="",
            available=False,
            error="pytesseract not installed. Run: pip install pytesseract",
        )

    try:
        # Get raw text
        raw_text = pytesseract.image_to_string(pil_img, lang="eng")

        # Get bounding boxes per word
        data = pytesseract.image_to_data(
            pil_img, lang="eng", output_type=pytesseract.Output.DICT
        )

        regions = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            if not word:
                continue
            conf = float(data["conf"][i])
            if conf < 0:
                continue
            regions.append(
                TextRegion(
                    text=word,
                    x=data["left"][i],
                    y=data["top"][i],
                    width=data["width"][i],
                    height=data["height"][i],
                    confidence=conf,
                )
            )

        words = [r.text for r in regions]
        return OCRResult(raw_text=raw_text.strip(), regions=regions, words=words)

    except Exception as exc:
        err = str(exc)
        # Provide helpful hint for missing Tesseract binary
        if "tesseract" in err.lower() and ("not found" in err.lower() or "is not recognized" in err.lower()):
            err = (
                "Tesseract binary not found. Install from "
                "https://github.com/UB-Mannheim/tesseract/wiki and set "
                "TESSERACT_CMD env var or add to PATH."
            )
        return OCRResult(raw_text="", available=False, error=err)
