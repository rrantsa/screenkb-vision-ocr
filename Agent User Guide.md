# screenkb — Agent User Guide

**screenkb** is a CLI tool that analyzes **any image** — screenshots, photos, scanned documents, and more — and returns structured JSON.
It uses OCR (Tesseract) + layout analysis (OpenCV) + computer vision (OpenCV faces/edges/MSER/scene) + a local knowledge base (SQLite).
No GPU. No internet required. Works on Windows, Linux, WSL.

> **screenkb v2.0.0+ handles all image types.** Photos of people, documents, landscapes, and objects are fully supported via the built-in CV Analyzer module. The pipeline auto-detects whether the image is a UI screenshot or a photo and classifies accordingly.

---

## Installation

### Requirements

- Python 3.10+
- Tesseract OCR binary

### Install Tesseract (Windows)

```bash
winget install UB-Mannheim.TesseractOCR
```

Auto-detected at `C:\Program Files\Tesseract-OCR\tesseract.exe`.

### Install Tesseract (Linux)

```bash
sudo apt install tesseract-ocr tesseract-ocr-fra
```

Or install without sudo (see `screenkb-vision-ocr` skill → `references/no-sudo-tesseract-install.md`).

### Install Python dependencies

```bash
cd screenkb/
pip install -e .
```

### Verify

```bash
screenkb --version
screenkb analyze screenshot.png
```

---

## Commands

### `analyze` — Classify any image

```bash
screenkb analyze <image>
screenkb analyze <image> --compact       # Agent-friendly (~200 tokens)
screenkb analyze <image> --extended      # Full data (~1700 tokens)
screenkb analyze <image> --output result.json
screenkb analyze <image> --verbose
screenkb analyze <image> --llm
```

**Output modes:**

| Flag | Token Size | Description |
|------|-----------|-------------|
| (default) | ~500 tokens | Standard: application, screen_type, description, confidence, visible_elements. Includes `photo_analysis` if the image is a photo. |
| `--compact` | ~200 tokens | Semantic: screen_type, summary, elements[], issues[] — ideal for AI agents |
| `--extended` | ~1700 tokens | Full data: OCR regions with x/y coords + layout regions + CV analysis + matched rules |
| `--debug` | full | Extended + fallback_reason + raw feature set |

**Default output for a UI screenshot:**
```json
{
  "application": "VS Code",
  "screen_type": "python_error",
  "description": "A Python exception (traceback) is visible.",
  "confidence": 0.92,
  "visible_elements": ["editor", "sidebar", "terminal"]
}
```

**Default output for a photo (auto-detects CV analysis):**
```json
{
  "application": "Unknown",
  "screen_type": "people_photo",
  "description": "A photo of one or more people.",
  "confidence": 0.85,
  "visible_elements": ["face_detected", "multiple_faces", "photo"],
  "photo_analysis": {
    "is_photo": true,
    "scene_type": "people_photo",
    "scene_confidence": 0.85,
    "face_count": 2,
    "skin_ratio": 18.5,
    "brightness": 145,
    "dominant_color": [120, 100, 80]
  }
}
```

**Extended output (`--extended`) includes `cv_analysis` field for all image types:**
```json
{
  "application": "VS Code",
  "screen_type": "python_error",
  "description": "...",
  "confidence": 0.92,
  "features": ["terminal", "traceback", "sidebar"],
  "ocr_count": 34,
  "region_count": 3,
  "ocr": [
    { "text": "ModuleNotFoundError", "x": 412, "y": 721, "width": 220, "height": 24, "confidence": 96.0 }
  ],
  "regions": [
    { "type": "sidebar", "x": 0, "y": 40, "width": 260, "height": 840, "area_ratio": 0.27 }
  ],
  "cv_analysis": {
    "is_photo": false,
    "scene_type": "ui_screenshot",
    "scene_confidence": 0.60,
    "face_count": 0,
    "skin_ratio": 0.0,
    "edge_ratio": 8.3,
    "mser_count": 42,
    "brightness": 210,
    "dominant_color": [240, 240, 240]
  },
  "matched_rules": [
    { "rule": "python_error", "score": 0.9, "matched_terms": ["traceback", "terminal"], "description": "Python exception" }
  ]
}
```

---

### `ocr` — Extract raw text with coordinates

```bash
screenkb ocr <image>
```

```json
{
  "word_count": 34,
  "text_regions": [
    { "text": "ModuleNotFoundError", "x": 412, "y": 721, "width": 220, "height": 24, "confidence": 96.0 },
    { "text": "main.py", "x": 88, "y": 54, "width": 64, "height": 18, "confidence": 93.0 }
  ]
}
```

---

### `layout` — Detect structure

```bash
screenkb layout <image>
```

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "region_count": 3,
  "region_labels": ["content", "sidebar", "terminal"],
  "regions": [
    { "type": "sidebar", "x": 0, "y": 40, "width": 260, "height": 840, "area_ratio": 0.27 },
    { "type": "terminal", "x": 260, "y": 680, "width": 1260, "height": 220, "area_ratio": 0.30 }
  ]
}
```

**Region types**: `sidebar`, `toolbar`, `menubar`, `statusbar`, `terminal`, `panel`, `content`

---

### `learn` — Teach the knowledge base

```bash
screenkb learn <image> screen_type=<label>
```

```bash
screenkb learn screenshot.png screen_type=login_form
screenkb learn screenshot.png screen_type=python_error
screenkb learn photocard.png screen_type=people_photo
screenkb learn scan.jpg screen_type=document
```

```json
{
  "status": "learned",
  "screen_type": "login_form",
  "features_saved": ["login_form", "sidebar"],
  "image_hash": "62f2c2945a714c32444ef4ff96d8d0c9"
}
```

Next time a similar image is analyzed, it gets a confidence boost.

---

### `explain` — Why was this classified?

```bash
screenkb explain <image>
```

```json
{
  "classification": "python_error",
  "confidence": 0.91,
  "reason": "Detected traceback, terminal.",
  "features_detected": ["traceback", "terminal", "sidebar"],
  "matched_rules": [
    { "rule": "python_error", "score": 0.9, "matched_terms": ["traceback", "terminal"] }
  ]
}
```

---

### `rules` — List all IF-THEN rules

```bash
screenkb rules
```

---

## Computer Vision (CV) Analysis — All Image Types

screenkb 2.0.0+ includes a **CV Analyzer** module (`screenkb/cv_analyzer.py`) that uses OpenCV to understand any image, not just UI screenshots.

### Pipeline (6 steps)

1. **OCR** (Tesseract) — text extraction
2. **Layout** (OpenCV) — UI regions (toolbar, sidebar, content)
3. **CV Analysis** (OpenCV) — faces, skin, edges, MSER, scene classification
4. **Knowledge base** — SQLite pattern matching
5. **Reasoning** — rule-based classification with CV fallback
6. **Output** — compact / extended / debug

### Scene types detected by CV analysis

| Type | Description | Trigger |
|------|-------------|---------|
| `people_photo` | Photo of people | Faces detected OR skin > 15% |
| `building_photo` | Photo of building/architecture | Warm tones + no faces + high MSER |
| `document` | Photo/scan of document | High brightness + MSER > 30 |
| `landscape` | Outdoor/landscape photo | Few MSER, moderate edges, high brightness |
| `object_photo` | Close-up of object | Very few contours, dark or blurry |
| `ui_screenshot` | UI screenshot (default) | None of the above |

### CV analysis fields (in `--extended` / `--debug` output)

| Field | Type | Description |
|-------|------|-------------|
| `is_photo` | bool | True if the image looks like a photo (not a UI screenshot) |
| `scene_type` | string | `people_photo`, `building_photo`, `document`, `landscape`, `object_photo`, or `ui_screenshot` |
| `scene_confidence` | float | 0.0–1.0 |
| `face_count` | int | Number of faces detected (Haar cascades) |
| `faces[]` | array | Array of `{x, y, width, height}` for each face |
| `skin_ratio` | float | % of image covered by skin-tone pixels (0.0–100.0) |
| `edge_intensity` | float | 0–255 average of Canny edge image |
| `edge_ratio` | float | % of pixels that are edges (0.0–100.0) |
| `contour_count` | int | Number of detected contours (after filtering) |
| `largest_contours[]` | array | Top 3 contours by area, with bounds |
| `mser_count` | int | MSER text-like region count (useful even without OCR) |
| `brightness` | integer | 0–255 average brightness |
| `dominant_color` | array | [B, G, R] average color |
| `features` | array | Feature labels for reasoning engine integration |

### In `--compact` mode

CV info is folded into the description:
- `"A photo of one or more people (2 personnes)."` (includes face count)
- `"A photo or scan of a document with text."`
- `"An outdoor or landscape photo."`

---

## Annotation Detection

Detects hand-drawn marks (circles, underlines, highlights) on images.

```bash
screenkb analyze <image> --annotations --annotation-colors=red,blue,green
```

Adds to the JSON output:
- `annotations[]` — list of detected marks with color, shape, position, confidence
- `annotated_elements[]` — which OCR text elements each annotation surrounds/points to

**Color presets:** red, blue, green, yellow, pink, orange

**Heuristic:** if >10 marks are found, the image likely has decorative elements (cards, art) — annotations are suppressed to avoid false positives.

---

## Workflow: Analyze → Present → Correct → Learn (Agent Mode)

This is the recommended workflow for AI agents analyzing images.

### Step 1 — Analyze (compact mode, agent-friendly)

```bash
screenkb analyze <image_path> --compact --verbose
```

Parse the JSON. `screen_type` tells you the image type. `confidence` tells you how reliable the classification is.

### Step 2 — Understand the image type

| `screen_type` | What to do next |
|---------------|-----------------|
| `people_photo` | Present face count, scene confidence. Do NOT look for text. |
| `document` | OCR is likely reliable — extract and post-process text. |
| `landscape` / `object_photo` | Present CV analysis (brightness, colors, contours). |
| `ui_screenshot` | Extract and post-process OCR text. Check for annotations if user indicated marks. |
| `unknown` | Confidence < 0.5 — ask user to clarify or use `screenkb learn`. |

### Step 3 — Get CV details (when OCR finds nothing)

When OCR returns <5 words with all confidence <50, switch to `--extended` mode:

```bash
screenkb analyze <image_path> --extended --verbose
```

Look at `cv_analysis`:
- `cv_analysis.is_photo` → True = it's a photo (people, landscape, object)
- `cv_analysis.scene_type` → specific type
- `cv_analysis.face_count` → number of detected faces
- If `people_photo`, present face count and skin ratio
- If `ui_screenshot` with no text, the image has icons/graphic text — try `--annotations`

**Do NOT invent text** — the CV module classifies scenes, it does not read text.

### Step 4 — Post-process OCR text (for UI screenshots)

Raw Tesseract JSON is not human-readable. Apply these transformations:

1. **Filter noise** — discard words with `confidence < 60` (or `< 75` for tokens ≤2 chars)
2. **Cluster by Y** — group words on the same line (Y gap tolerance ~25px)
3. **Sort by X** — left-to-right reading order within each line
4. **Reconstruct sentences** — join clustered words with spaces
5. **Detect language** — French markers (`de`, `le`, `la`, `pour`, `avec`) or Malagasy markers (`ny`, `tsy`, `dia`)
6. **Infer misreads** — `AL` → `AI`/`Ar` (Ariary), `Contréle` → `Contrôle`, single-char noise → discard

### Step 5 — Check for annotations (when applicable)

```bash
screenkb analyze <image_path> --annotations
```

Use when the user says they circled, highlighted, or marked something.

### Step 6 — Flag uncertainties

- **App names:** If nav-bar logo text has confidence < 80, DO NOT assert an app name. Say the function instead.
- **Proper names:** Names with confidence < 90 are frequently misread — present as tentative.
- **Low confidence:** If `confidence < 0.5`, flag it and offer to train with `screenkb learn`.

### Step 7 — Accept user corrections gracefully

1. Acknowledge the correction
2. Update your understanding
3. Log the correction for this session
4. Optionally fix the KB with `screenkb learn <image> screen_type=<correct_label>`

### Step 8 — Train the knowledge base

```bash
screenkb learn <image_path> screen_type=<correct_label>
```

This saves the image's feature fingerprint so future similar images get a confidence boost.

---

## How to Use as an AI Agent Tool

### Python integration

```python
import subprocess, json

# Analyze with compact output
result = json.loads(
    subprocess.check_output(
        ["screenkb", "analyze", "image.png", "--compact"]
    )
)
screen_type = result["screen_type"]   # e.g. "login_form", "people_photo"
confidence  = result["confidence"]    # e.g. 0.80
```

### Get coordinates (when needed)

```python
result = json.loads(
    subprocess.check_output(
        ["screenkb", "analyze", "image.png", "--extended"]
    )
)
for region in result["ocr"]:
    print(f"{region['text']} at ({region['x']}, {region['y']})")

# Check CV analysis
if "cv_analysis" in result:
    cv = result["cv_analysis"]
    print(f"Scene: {cv['scene_type']}, Faces: {cv['face_count']}")
```

### LLM enhancement (optional)

```bash
screenkb analyze screenshot.png --llm --llm-backend deepseek
screenkb analyze screenshot.png --llm --llm-backend openai
screenkb analyze screenshot.png --llm --llm-backend ollama
```

The LLM receives only extracted features + OCR text — never the raw image.

---

## Confidence Thresholds (recommended)

| Confidence | Meaning | Suggested action |
|-----------|---------|-----------------|
| ≥ 0.90 | High | Act on it directly |
| 0.70–0.89 | Medium | Act + log for review |
| 0.50–0.69 | Low | Confirm with `explain` |
| < 0.50 | Unknown | Teach with `learn` |

---

## Known screen_type values

| screen_type | Description |
|------------|-------------|
| `python_error` | Python traceback in terminal |
| `javascript_error` | JS runtime error |
| `http_error` | HTTP 4xx/5xx page |
| `stack_trace` | Generic stack trace |
| `login_form` | Login / authentication form |
| `dashboard` | Application dashboard |
| `code_editor` | Code editor view (VSCode etc.) |
| `vscode` | Visual Studio Code |
| `terminal` | Terminal or CLI window |
| `api_response` | JSON/API response display |
| `test_runner` | Test runner output |
| `form` | Generic web form |
| `browser` | Web browser window |
| `kanban` | Kanban board or card-based task view |
| `statistics` | Statistics or analytics dashboard |
| `people_photo` | Photo of one or more people |
| `building_photo` | Photo of a building or architecture |
| `document` | Photo or scan of a document |
| `landscape` | Outdoor or landscape photo |
| `object_photo` | Close-up photo of an object |
| `ui_screenshot` | Application or UI screenshot |
| `unknown` | Could not be reliably classified |

Custom types can be added via `screenkb learn`.

---

## Knowledge Base Location

| Platform | Path |
|----------|------|
| Windows | `%APPDATA%\screenkb\screenkb.db` |
| Linux | `~/.config/screenkb/screenkb.db` |

Override with `SCREENKB_DB` env var.

---

## Supported Image Formats

PNG, JPG, JPEG, BMP, WEBP

---

## Performance

Target: **< 3 seconds** per 1080p image on standard hardware (4 cores, 8 GB RAM).
CV analysis adds ~0.5–1.5s depending on image size and face detection.
