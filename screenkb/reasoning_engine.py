"""Reasoning Engine — score IF-THEN rules against extracted features."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING

from .ocr_engine import OCRResult
from .layout_analyzer import LayoutResult
from .feature_extractor import extract_features, detect_application
from .knowledge_base import (
    get_all_rules,
    find_matching_patterns,
    lookup_correction,
)

if TYPE_CHECKING:
    from .cv_analyzer import CVResult


@dataclass
class RuleMatch:
    screen_type: str
    score: float
    matched_required: List[str]
    matched_bonus: List[str]
    description: str


@dataclass
class ReasoningResult:
    application: str
    screen_type: str
    description: str
    confidence: float
    visible_elements: List[str]
    rule_matches: List[RuleMatch] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    ocr_data: Optional[Any] = field(default=None)
    layout_data: Optional[Any] = field(default=None)
    cv_data: Optional[Any] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Compact output — classification only."""
        d = {
            "application": self.application,
            "screen_type": self.screen_type,
            "description": self.description,
            "confidence": round(self.confidence, 2),
            "visible_elements": self.visible_elements,
        }
        if self.cv_data and self.cv_data.is_photo:
            d["photo_analysis"] = self.cv_data.to_dict()
        return d

    def to_full_dict(self) -> Dict[str, Any]:
        """Extended output — classification + features + full OCR regions (with coords) + layout regions + matched rules."""
        d = self.to_dict()
        d["features"] = self.features
        d["ocr_count"] = len(self.ocr_data.words) if self.ocr_data else 0
        d["region_count"] = len(self.layout_data.regions) if self.layout_data else 0
        d["ocr"] = [r.to_dict() for r in self.ocr_data.regions] if self.ocr_data else []
        d["regions"] = [r.to_dict() for r in self.layout_data.regions] if self.layout_data else []
        if self.cv_data and self.cv_data.available:
            d["cv_analysis"] = self.cv_data.to_dict()
        d["matched_rules"] = [
            {
                "rule": m.screen_type,
                "score": round(m.score, 3),
                "matched_terms": m.matched_required + m.matched_bonus,
                "description": m.description,
            }
            for m in self.rule_matches
        ]
        return d

    def to_debug_dict(self) -> Dict[str, Any]:
        """Debug output — everything + fallback_reason + issues list."""
        d = self.to_full_dict()
        if self.confidence >= 0.99 and not self.rule_matches:
            d["fallback_reason"] = "Correction cache hit (exact image match from screenkb learn)."
        elif self.rule_matches:
            d["fallback_reason"] = f"Rule '{self.rule_matches[0].screen_type}' matched with score {self.rule_matches[0].score:.2f}."
        else:
            d["fallback_reason"] = "No rules matched — classified by fallback heuristic based on features."
        d["raw_feature_set"] = sorted(self.features)
        return d

    def to_verbose_dict(self) -> Dict[str, Any]:
        return self.to_full_dict()


# Confidence modifiers
_BONUS_WEIGHT = 0.05     # each matched bonus feature adds this
_PATTERN_BOOST_CAP = 0.2  # max boost from learned patterns


def run_reasoning(
    ocr: OCRResult,
    layout: LayoutResult,
    conn,
    image_hash: Optional[str] = None,
    cv_result: Optional[Any] = None,
) -> ReasoningResult:
    """Run full reasoning pipeline and return structured result."""

    # 1. Check if we have a stored correction for this exact image
    if image_hash and conn:
        correction = lookup_correction(conn, image_hash)
        if correction:
            features = extract_features(ocr, layout)
            if cv_result and cv_result.available:
                features.extend(cv_result.features)
            application = detect_application(ocr)
            return ReasoningResult(
                application=application,
                screen_type=correction,
                description=_describe(correction, application, features),
                confidence=0.99,
                visible_elements=_visible_elements(layout, features),
                features=features,
                ocr_data=ocr,
                layout_data=layout,
                cv_data=cv_result,
            )

    # 2. Extract features
    features = extract_features(ocr, layout)
    # Add CV features if available
    if cv_result and cv_result.available:
        features.extend(cv_result.features)
    application = detect_application(ocr)
    feature_set = set(features)

    # 3. Score rules
    rules = get_all_rules(conn) if conn else []
    matches: List[RuleMatch] = []

    for rule in rules:
        required = rule["required_features"]
        bonus = rule["bonus_features"]

        # All required features must be present
        matched_req = [f for f in required if f in feature_set]
        if len(matched_req) < len(required):
            continue  # rule doesn't fire

        matched_bonus = [f for f in bonus if f in feature_set]
        score = rule["base_confidence"] + len(matched_bonus) * _BONUS_WEIGHT
        score = min(score, 0.97)  # cap below 1.0

        matches.append(RuleMatch(
            screen_type=rule["screen_type"],
            score=score,
            matched_required=matched_req,
            matched_bonus=matched_bonus,
            description=rule["description"],
        ))

    # 4. Apply learned pattern boosts
    if conn:
        pattern_boosts: Dict[str, float] = {}
        for screen_type, boost in find_matching_patterns(conn, features):
            pattern_boosts[screen_type] = min(
                pattern_boosts.get(screen_type, 0.0) + boost,
                _PATTERN_BOOST_CAP,
            )
        for m in matches:
            if m.screen_type in pattern_boosts:
                m.score = min(m.score + pattern_boosts[m.screen_type], 0.99)

    # 5. Sort and pick winner
    matches.sort(key=lambda m: m.score, reverse=True)

    if matches:
        winner = matches[0]
        screen_type = winner.screen_type
        confidence = winner.score
    else:
        # No rule fired — fallback to best-guess from features
        screen_type, confidence = _fallback(features, application)

    description = _describe(screen_type, application, features)
    visible = _visible_elements(layout, features)

    return ReasoningResult(
        application=application,
        screen_type=screen_type,
        description=description,
        confidence=confidence,
        visible_elements=visible,
        rule_matches=matches,
        features=features,
        ocr_data=ocr,
        layout_data=layout,
        cv_data=cv_result,
    )


def _fallback(features: List[str], application: str) -> Tuple[str, float]:
    """When no rules fire, make a best-effort guess."""
    # CV-based fallbacks (from photo analysis)
    if "scene_people_photo" in features:
        return "people_photo", 0.70
    if "scene_building_photo" in features:
        return "building_photo", 0.65
    if "scene_document" in features:
        return "document", 0.65
    if "scene_landscape" in features:
        return "landscape", 0.55
    if "scene_object" in features:
        return "object_photo", 0.50

    # Original fallbacks
    if "terminal" in features:
        return "terminal", 0.45
    if "editor" in features:
        return "code_editor", 0.40
    if "browser" in features:
        return "browser", 0.35
    if "login_form" in features:
        return "login_form", 0.50
    if application != "Unknown":
        return application.lower().replace(" ", "_"), 0.30
    return "unknown", 0.10


def _describe(screen_type: str, application: str, features: List[str]) -> str:
    """Generate a human-readable one-line description."""
    templates = {
        "python_error": "A Python exception (traceback) is visible.",
        "javascript_error": "A JavaScript runtime error is displayed.",
        "http_error": "An HTTP error page is shown.",
        "stack_trace": "A stack trace is visible in the output.",
        "login_form": f"A login or authentication form is displayed{' in ' + application if application != 'Unknown' else ''}.",
        "dashboard": f"An application dashboard is open{' in ' + application if application != 'Unknown' else ''}.",
        "code_editor": f"A code editor view is open{' in ' + application if application != 'Unknown' else ''}.",
        "vscode": "Visual Studio Code is open with a code editor view.",
        "terminal": "A terminal or command-line interface is displayed.",
        "api_response": "An API response (likely JSON) is shown.",
        "test_runner": "A test runner output is displayed.",
        "form": "A web form is visible.",
        "browser": f"A web browser window is open{' in ' + application if application != 'Unknown' else ''}.",
        "people_photo": "A photo of one or more people.",
        "building_photo": "A photo of a building or architecture.",
        "document": "A photo or scan of a document with text.",
        "landscape": "An outdoor or landscape photo.",
        "object_photo": "A close-up photo of an object.",
        "ui_screenshot": "An application or UI screenshot.",
        "unknown": "The image content could not be reliably classified.",
    }
    desc = templates.get(screen_type)
    if desc:
        return desc
    # Generic fallback
    return f"A {screen_type.replace('_', ' ')} screen is displayed."


def _visible_elements(layout: LayoutResult, features: List[str]) -> List[str]:
    """Produce a deduplicated list of visible UI elements."""
    elements = set()
    # From layout
    for label in (layout.region_labels or []):
        elements.add(label)
    # From features — only structural ones worth reporting
    structural = {
        "editor", "terminal", "sidebar", "toolbar", "menubar", "statusbar",
        "panel", "form", "modal", "table", "navigation", "browser",
        "face_detected", "multiple_faces", "skin_present", "skin_abundant",
        "photo", "ui_screenshot",
    }
    for f in features:
        if f in structural:
            elements.add(f)
    return sorted(elements)


def compute_image_hash(pil_img) -> str:
    """Compute a fast perceptual hash (MD5 of resized thumbnail bytes)."""
    thumb = pil_img.copy()
    thumb.thumbnail((64, 64))
    return hashlib.md5(thumb.tobytes()).hexdigest()
