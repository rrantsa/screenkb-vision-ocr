"""Compact Output — semantic element extraction for agent-friendly, low-token output."""

from __future__ import annotations

from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .reasoning_engine import ReasoningResult
    from .ocr_engine import TextRegion
    from .layout_analyzer import LayoutResult

_MIN_CONFIDENCE = 60.0  # OCR regions below this are noise
_MIN_CONFIDENCE_SHORT = 75.0  # short tokens (≤2 chars) need higher confidence — likely UI icons

# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def build_compact_output(result: "ReasoningResult") -> Dict[str, Any]:
    """Return a token-efficient semantic output (~180-260 tokens vs ~1700+ for --extended)."""
    ocr = result.ocr_data
    layout = result.layout_data

    # Filter noise — short tokens need higher confidence (likely icon/artifact OCR)
    clean = [
        r for r in (ocr.regions if ocr else [])
        if len(r.text.strip()) > 1 and (
            r.confidence >= _MIN_CONFIDENCE_SHORT if len(r.text.strip()) <= 2
            else r.confidence >= _MIN_CONFIDENCE
        )
    ]

    # Cluster adjacent text into lines
    clusters = _cluster_by_y(clean, gap=25)

    # Classify each line into a semantic element
    img_h = layout.image_height if layout else 1
    elements = []
    for cluster in clusters:
        # Sort within cluster by X position for correct reading order
        cluster_sorted = sorted(cluster, key=lambda r: r.x)
        text = " ".join(r.text for r in cluster_sorted).strip()
        if len(text) < 2:
            continue
        elem_type = _classify_element(text, cluster_sorted, img_h)
        if elem_type != "noise":
            elements.append({"type": elem_type, "text": text})

    # Build summary
    summary = _generate_summary(result, elements, ocr)

    # Detect issues
    issues = _build_issues(result)

    out: Dict[str, Any] = {
        "screen_type": result.screen_type,
        "confidence": round(result.confidence, 2),
        "application": result.application,
        "summary": summary,
        "elements": elements,
    }
    if issues:
        out["issues"] = issues
    return out


# ──────────────────────────────────────────────────────────────────────────────
# OCR clustering
# ──────────────────────────────────────────────────────────────────────────────

def _cluster_by_y(regions: List, gap: int = 25) -> List[List]:
    if not regions:
        return []
    sorted_r = sorted(regions, key=lambda r: (r.y, r.x))
    clusters: List[List] = [[sorted_r[0]]]
    for r in sorted_r[1:]:
        last = clusters[-1][-1]
        # same line if Y overlap or Y gap ≤ gap
        if r.y <= last.y + last.height + gap:
            clusters[-1].append(r)
        else:
            clusters.append([r])
    return clusters


# ──────────────────────────────────────────────────────────────────────────────
# Element classification
# ──────────────────────────────────────────────────────────────────────────────

_BUTTON_KW = [
    "se connecter", "connexion", "login", "sign in", "submit",
    "enregistrer", "save", "cancel", "annuler", "confirmer", "confirm",
    "valider", "validate", "continuer", "continue", "suivant", "next",
]
_LINK_KW = [
    "oubli", "forgot", "reset", "continuer avec", "lien",
    "mot de passe oubli", "password forgot", "sign up", "register",
]
_INPUT_KW = [
    "email", "adresse", "mot de passe", "password", "username",
    "identifiant", "nom", "prénom", "first name", "last name",
    "phone", "téléphone",
]
_TITLE_KW = [
    "bienvenue", "welcome", "bonjour", "hello", "sign in to",
    "connexion à", "interface", "administration", "tableau de bord",
]
_SSO_KW = ["microsoft", "google", "github", "facebook", "apple", "twitter", "linkedin"]
_TOOLBAR_KW = [
    "netlify", "deploy preview", "collaborate", "drawer", "preview",
    "log in to", "vercel", "heroku", "staging",
]
_ERROR_KW = [
    "error", "erreur", "traceback", "exception", "warning", "avertissement",
    "not found", "forbidden", "unauthorized",
]
_NAV_KW = [
    "dashboard", "tableau de bord", "settings", "paramètres", "logout",
    "déconnexion", "home", "accueil", "menu", "navigation", "search", "recherche",
]


def _classify_element(text: str, cluster: List, img_h: int) -> str:
    tl = text.lower()

    # Toolbar — bottom 15% of screen
    avg_y = sum(r.y for r in cluster) / len(cluster)
    if img_h > 0 and avg_y / img_h > 0.85:
        return "toolbar"

    if any(kw in tl for kw in _TOOLBAR_KW):
        return "toolbar"
    if any(kw in tl for kw in _ERROR_KW):
        return "error"
    if any(kw in tl for kw in _SSO_KW):
        return "sso_button"
    if any(kw in tl for kw in _BUTTON_KW):
        return "button"
    if any(kw in tl for kw in _LINK_KW):
        return "link"
    if any(kw in tl for kw in _INPUT_KW):
        return "input"
    if any(kw in tl for kw in _TITLE_KW):
        return "title"
    if any(kw in tl for kw in _NAV_KW):
        return "nav"

    # Short single-char or punctuation-only → noise
    if len(text.strip(".,;:!?-_/\\|*#@^~`<>()[]{}\"' ")) < 2:
        return "noise"

    return "text"


# ──────────────────────────────────────────────────────────────────────────────
# Summary generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_summary(result: "ReasoningResult", elements: List[Dict], ocr) -> str:
    raw = (ocr.raw_text if ocr else "").lower()

    # Detect language
    lang = ""
    fr_markers = ["de ", "le ", "la ", "les ", "vous ", "dans ", "avec ", "mot de passe"]
    if sum(1 for m in fr_markers if m in raw) >= 2:
        lang = "French"

    # Build key observations
    elem_types = [e["type"] for e in elements]
    has_input = "input" in elem_types
    has_button = "button" in elem_types
    has_sso = "sso_button" in elem_types
    has_toolbar = "toolbar" in elem_types
    has_error = "error" in elem_types

    # Screen-type-aware templates
    st = result.screen_type
    parts = []

    if lang:
        parts.append(lang)

    if st == "login_form":
        desc = "login page"
        if result.application != "Unknown":
            desc = f"{result.application} login page"
        if has_input:
            desc += " with email/password fields"
        if has_sso:
            sso_texts = [e["text"] for e in elements if e["type"] == "sso_button"]
            desc += f" and {', '.join(sso_texts[:2])} SSO"
        if has_toolbar:
            toolbar_texts = [e["text"] for e in elements if e["type"] == "toolbar"]
            desc += f" (preview: {toolbar_texts[0][:40]})" if toolbar_texts else " (preview environment)"
        parts.append(desc + ".")
    elif st == "dashboard":
        parts.append(f"Application dashboard{' in ' + result.application if result.application != 'Unknown' else ''}.")
    elif st in ("python_error", "javascript_error", "stack_trace"):
        parts.append(f"{st.replace('_', ' ').title()} displayed in {'terminal' if 'terminal' in result.features else 'editor'}.")
    elif st in ("people_photo", "document", "landscape", "object_photo", "building_photo"):
        # CV-based description — add face count if available
        if result.cv_data and result.cv_data.face_count:
            face_info = f" ({result.cv_data.face_count} personne{'s' if result.cv_data.face_count > 1 else ''})"
            desc = result.description.replace(".", "") + face_info + "."
        else:
            desc = result.description
        parts.append(desc)
    else:
        parts.append(result.description)

    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Issue detection
# ──────────────────────────────────────────────────────────────────────────────

def _build_issues(result: "ReasoningResult") -> List[str]:
    issues = []

    if not result.rule_matches:
        if result.confidence >= 0.99:
            pass  # correction cache hit — not an issue
        else:
            issues.append("No deterministic rule matched; classified by fallback heuristic.")

    if result.layout_data:
        for r in result.layout_data.regions:
            if r.label == "sidebar" and r.area_ratio > 0.4:
                issues.append(
                    f"Region [{r.x},{r.y}] covers {r.area_ratio*100:.0f}% of screen "
                    f"but was classified as sidebar (likely main content area)."
                )

    if result.ocr_data:
        noise_count = sum(
            1 for r in result.ocr_data.regions
            if r.confidence < _MIN_CONFIDENCE or len(r.text.strip()) <= 1
        )
        if noise_count > 3:
            issues.append(f"{noise_count} noisy OCR tokens (confidence < 50 or single char).")

    if result.confidence < 0.5:
        issues.append(
            f"Low confidence ({result.confidence:.2f}). "
            "Run: screenkb learn screenshot.png screen_type=<correct_label>"
        )

    return issues
