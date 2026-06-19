"""screenkb CLI — main entry point."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .image_loader import load_image, ImageLoadError
from .ocr_engine import run_ocr
from .layout_analyzer import analyze_layout
from .feature_extractor import extract_features, detect_application
from .reasoning_engine import run_reasoning, compute_image_hash
from .knowledge_base import init_db, save_correction, get_all_rules
from .annotation_detector import detect_annotations
from .cv_analyzer import analyze_cv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pipeline(image_path: str, verbose: bool = False):
    """Run full pipeline and return (pil_img, cv_img, ocr, layout, cv, conn, hash)."""
    t0 = time.monotonic()

    if verbose:
        click.echo(f"[1/6] Loading image: {image_path}", err=True)
    pil_img, cv_img = load_image(image_path)
    if verbose:
        click.echo(f"      Size: {pil_img.size[0]}x{pil_img.size[1]} px", err=True)

    if verbose:
        click.echo("[2/6] Running OCR (Tesseract)...", err=True)
    ocr = run_ocr(pil_img)
    if verbose:
        if ocr.available:
            click.echo(f"      Words detected: {len(ocr.words)}", err=True)
        else:
            click.echo(f"      OCR unavailable: {ocr.error}", err=True)

    if verbose:
        click.echo("[3/6] Analyzing layout (OpenCV)...", err=True)
    layout = analyze_layout(cv_img)
    if verbose:
        if layout.available:
            click.echo(f"      Regions found: {len(layout.regions)} ({', '.join(layout.region_labels)})", err=True)
        else:
            click.echo(f"      Layout unavailable: {layout.error}", err=True)

    if verbose:
        click.echo("[4/6] Running CV analysis (OpenCV — faces, edges, MSER)...", err=True)
    cv_result = analyze_cv(cv_img)
    if verbose:
        if cv_result.available:
            parts = []
            if cv_result.face_count:
                parts.append(f"{cv_result.face_count} face(s)")
            parts.append(f"scene={cv_result.scene_type}")
            parts.append(f"edges={cv_result.edge_ratio:.1f}%")
            parts.append(f"mser={cv_result.mser_count}")
            click.echo(f"      CV: {', '.join(parts)}", err=True)
        else:
            click.echo(f"      CV unavailable: {cv_result.error}", err=True)

    if verbose:
        click.echo("[5/6] Connecting to knowledge base...", err=True)
    conn = init_db()

    if verbose:
        click.echo("[6/6] Running reasoning engine...", err=True)
    image_hash = compute_image_hash(pil_img)

    elapsed = time.monotonic() - t0
    if verbose:
        click.echo(f"      Pipeline ready in {elapsed:.2f}s", err=True)

    return pil_img, cv_img, ocr, layout, cv_result, conn, image_hash

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version=__version__, prog_name="screenkb")
def cli():
    """screenkb — Analyze screenshots and produce structured JSON output.

    Uses OCR + layout analysis + knowledge base rules. No GPU required.
    """


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Write JSON to this file path.")
@click.option("--verbose", "-v", is_flag=True, help="Show pipeline steps on stderr.")
@click.option("--compact", "-c", is_flag=True, help="Semantic output: {screen_type, summary, elements[]} — low-token, agent-friendly (~200 tokens).")
@click.option("--extended", "-e", is_flag=True, help="Full coords: standard + OCR regions + layout regions + matched rules (~1700 tokens).")
@click.option("--debug", "-d", is_flag=True, help="Debug mode: --extended + fallback_reason + raw feature set.")
@click.option("--llm", is_flag=True, help="Enhance result with an LLM (DeepSeek/OpenAI/Ollama).")
@click.option("--llm-backend", default=None, help="LLM backend: deepseek | openai | ollama.")
@click.option("--annotations", "-a", is_flag=True, help="Detect hand-drawn annotations (circles, underlines, highlights) on the image. Adds 'annotations' and 'annotated_elements' to output.")
@click.option("--annotation-colors", default="red,blue,green", help="Comma-separated colors to detect (default: red,blue,green).")
def analyze(image_path: str, output: Optional[str], verbose: bool, compact: bool,
            extended: bool, debug: bool, llm: bool, llm_backend: Optional[str],
            annotations: bool, annotation_colors: str):
    """Analyze a screenshot and output structured JSON.

    Three output modes (pick one):

    \b
        (default)   Standard PRD output: {application, screen_type, description, confidence, visible_elements}
        --compact   Semantic agent output: {screen_type, summary, elements[], issues[]}  (~200 tokens)
        --extended  Full coords: standard + OCR regions with x/y + layout regions (~1700 tokens)
        --debug     --extended + fallback_reason + raw feature set (for QA investigation)

    \b
    Example:
        screenkb analyze screenshot.png
        screenkb analyze screenshot.png --compact
        screenkb analyze screenshot.png --extended
        screenkb analyze screenshot.png --debug
        screenkb analyze screenshot.png --compact --output result.json
        screenkb analyze screenshot.png --llm
    """
    try:
        pil_img, cv_img, ocr, layout, cv_result, conn, image_hash = _load_pipeline(image_path, verbose)
    except ImageLoadError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    result = run_reasoning(ocr, layout, conn, image_hash, cv_result=cv_result)
    conn.close()

    # Detect annotations if requested
    if annotations:
        if verbose:
            click.echo("[Annotations] Detecting hand-drawn marks (OpenCV)...", err=True)
        colors_list = [c.strip() for c in annotation_colors.split(",") if c.strip()]
        ocr_regions = [r.to_dict() for r in ocr.regions] if ocr and ocr.regions else []
        # Scale: cv_img dimensions vs KB dimensions (screenkb normalizes to 1920x1080)
        img_h, img_w = cv_img.shape[:2]
        kb_w, kb_h = (pil_img.size[0], pil_img.size[1])
        scale_x = img_w / max(kb_w, 1)
        scale_y = img_h / max(kb_h, 1)
        ann_result = detect_annotations(cv_img, ocr_regions, colors=colors_list,
                                        scale_kb_to_img=(scale_x, scale_y))
        if verbose:
            if ann_result.available:
                click.echo(f"      Annotations found: {len(ann_result.annotations)}", err=True)
                if ann_result.annotated_elements:
                    click.echo(f"      Elements annotated: {len(ann_result.annotated_elements)}", err=True)
            else:
                click.echo(f"      Annotation detection unavailable: {ann_result.error}", err=True)

        # If too many annotations, it's likely decorative elements, not real annotations
        if len(ann_result.annotations) > 10:
            if verbose:
                click.echo(f"      Too many ({len(ann_result.annotations)}), likely decorative — suppressing output", err=True)
            ann_result.annotations = []
            ann_result.annotated_elements = []

    if llm:
        if verbose:
            click.echo("[LLM] Enhancing result...", err=True)
        try:
            from .llm_enhancer import enhance_with_llm
            enhanced = enhance_with_llm(
                features=result.features,
                ocr_text=ocr.raw_text,
                layout_labels=layout.region_labels,
                current_result=result.to_dict(),
                backend=llm_backend,
            )
            base = result.to_dict()
            base.update({k: v for k, v in enhanced.items() if v})
            output_data = base
        except Exception as e:
            click.echo(f"Warning: LLM enhancement failed: {e}", err=True)
            output_data = _pick_output(result, compact, extended, debug)
    else:
        output_data = _pick_output(result, compact, extended, debug)

    # Append annotation data if detected
    if annotations and 'ann_result' in dir() and ann_result.available:
        output_data["annotations"] = [a.to_dict() for a in ann_result.annotations]
        if ann_result.annotated_elements:
            output_data["annotated_elements"] = ann_result.annotated_elements

    json_str = json.dumps(output_data, indent=2, ensure_ascii=False)

    if output:
        Path(output).write_text(json_str, encoding="utf-8")
        click.echo(json.dumps({"status": "ok", "output": output}, indent=2))
    else:
        click.echo(json_str)


def _pick_output(result, compact: bool, extended: bool, debug: bool):
    if debug:
        return result.to_debug_dict()
    if extended:
        return result.to_full_dict()
    if compact:
        from .compact_output import build_compact_output
        return build_compact_output(result)
    return result.to_dict()


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def ocr(image_path: str):
    """Extract raw OCR text from a screenshot.

    \b
    Example:
        screenkb ocr screenshot.png
    """
    try:
        pil_img, _ = load_image(image_path)
    except ImageLoadError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    result = run_ocr(pil_img)

    if not result.available:
        click.echo(f"OCR unavailable: {result.error}", err=True)
        sys.exit(1)

    output_data = {
        "word_count": len(result.words),
        "text_regions": [r.to_dict() for r in result.regions],
    }
    click.echo(json.dumps(output_data, indent=2, ensure_ascii=False))


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def layout(image_path: str):
    """Show detected layout regions in a screenshot.

    \b
    Example:
        screenkb layout screenshot.png
    """
    try:
        _, cv_img = load_image(image_path)
    except ImageLoadError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    result = analyze_layout(cv_img)

    if not result.available:
        click.echo(f"Layout analysis unavailable: {result.error}", err=True)
        sys.exit(1)

    click.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("assignments", nargs=-1, required=True)
def learn(image_path: str, assignments: tuple):
    """Store a correction in the knowledge base.

    ASSIGNMENTS: key=value pairs, e.g. screen_type=python_error

    \b
    Example:
        screenkb learn screenshot.png screen_type=login_form
        screenkb learn screenshot.png screen_type=python_error
    """
    params = {}
    for a in assignments:
        if "=" not in a:
            click.echo(f"Error: invalid assignment '{a}' — expected key=value", err=True)
            sys.exit(1)
        k, v = a.split("=", 1)
        params[k.strip()] = v.strip()
    screen_type = params.get("screen_type")
    if not screen_type:
        click.echo("Error: 'screen_type' required (e.g. screen_type=login_form)", err=True)
        sys.exit(1)
    try:
        pil_img, cv_img = load_image(image_path)
    except ImageLoadError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    ocr_result = run_ocr(pil_img)
    layout_result = analyze_layout(cv_img)
    cv_result = analyze_cv(cv_img)
    features = extract_features(ocr_result, layout_result)
    if cv_result.available:
        features.extend(cv_result.features)
    image_hash = compute_image_hash(pil_img)

    conn = init_db()
    save_correction(conn, image_hash, screen_type, features)
    conn.close()

    click.echo(json.dumps({
        "status": "learned",
        "screen_type": screen_type,
        "features_saved": features,
        "image_hash": image_hash,
    }, indent=2))


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def explain(image_path: str):
    """Show which KB rules matched for a screenshot.

    \b
    Example:
        screenkb explain screenshot.png
    """
    try:
        pil_img, cv_img = load_image(image_path)
    except ImageLoadError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    ocr_result = run_ocr(pil_img)
    layout_result = analyze_layout(cv_img)
    cv_result = analyze_cv(cv_img)
    features = extract_features(ocr_result, layout_result)
    if cv_result.available:
        features.extend(cv_result.features)
    image_hash = compute_image_hash(pil_img)

    conn = init_db()
    # Pass image_hash=None to bypass correction cache — explain must show actual rule matches
    result = run_reasoning(ocr_result, layout_result, conn, image_hash=None)
    conn.close()

    # Build human-readable reason from top matched terms
    top_terms = []
    for m in result.rule_matches[:3]:
        top_terms.extend(m.matched_required + m.matched_bonus)
    seen = set()
    unique_terms = [t for t in top_terms if not (t in seen or seen.add(t))]
    reason = (
        f"Detected {', '.join(unique_terms[:4])}." if unique_terms
        else "No matching rules — classified by fallback heuristic."
    )

    output = {
        "classification": result.screen_type,
        "confidence": round(result.confidence, 3),
        "reason": reason,
        "features_detected": features,
        "matched_rules": [
            {
                "rule": m.screen_type,
                "score": round(m.score, 3),
                "matched_terms": m.matched_required + m.matched_bonus,
            }
            for m in result.rule_matches
        ],
    }

    click.echo(json.dumps(output, indent=2, ensure_ascii=False))


@cli.command("rules")
def list_rules():
    """List all IF-THEN rules in the knowledge base."""
    conn = init_db()
    rules = get_all_rules(conn)
    conn.close()
    click.echo(json.dumps(rules, indent=2, ensure_ascii=False))


def main():
    cli()


if __name__ == "__main__":
    main()
