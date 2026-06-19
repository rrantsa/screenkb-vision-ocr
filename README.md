# screenkb

**Analyze any image** — screenshots, photos, scanned documents — with OCR + computer vision + layout detection + a rule-based knowledge base. Produces structured JSON output. No GPU required. Fully offline.

## Features

- **OCR** (Tesseract) — text extraction with coordinates
- **Layout detection** (OpenCV) — panels, toolbars, sidebars, terminal panes
- **Computer Vision analysis** (OpenCV) — faces, skin, edges, MSER, scene classification
- **Annotation detection** — hand-drawn circles, underlines, highlights in color
- **Knowledge base** (SQLite) — learnable IF-THEN rules + pattern matching
- **All image types** — not just UI screenshots. Photos of people, documents, landscapes, objects are auto-detected and classified.
- **Compact output** (~200 tokens) — optimized for AI agent consumption

## Pipeline

```
Image Path
  → Image Loader       (Pillow — PNG/JPG/BMP/WEBP, normalize+resize)
  → OCR Engine         (pytesseract — text regions with coordinates)
  → Layout Analyzer    (OpenCV — panels, toolbars, sidebars, terminal)
  → CV Analyzer        (OpenCV — faces, skin, edges, MSER, scene type)
  → Feature Extractor  (keyword matching → feature list)
  → KB Matcher         (SQLite rules + learned patterns)
  → Reasoning Engine   (IF-THEN scoring + CV fallback → winner selection)
  → JSON Output
```

## Scene types (CV auto-detection)

| Type | Example |
|------|---------|
| `people_photo` | Group photo, selfie, portrait |
| `building_photo` | Architecture, building exterior |
| `document` | Photo of a paper, ID card, scan |
| `landscape` | Outdoor scenery, sky, nature |
| `object_photo` | Product photo, close-up of an object |
| `ui_screenshot` | Desktop/mobile app screen |

## Requirements

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (binary, must be installed separately)

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

On Windows, install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki  
Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe` (auto-detected).  
Override via env var if installed elsewhere:
```powershell
$env:TESSERACT_CMD = "D:\tools\Tesseract-OCR\tesseract.exe"
```

## Usage

### analyze — full pipeline → JSON

```bash
screenkb analyze screenshot.png                 # Default: ~500 tokens
screenkb analyze screenshot.png --compact       # Agent-friendly: ~200 tokens
screenkb analyze screenshot.png --extended      # Full data: ~1700 tokens
screenkb analyze screenshot.png --verbose       # Show pipeline steps
screenkb analyze screenshot.png --output result.json
screenkb analyze screenshot.png --llm           # LLM enhancement
screenkb analyze screenshot.png --llm --llm-backend deepseek
screenkb analyze screenshot.png --annotations   # Detect hand-drawn marks
```

### ocr — raw text extraction

```bash
screenkb ocr screenshot.png
```

### layout — detected UI regions

```bash
screenkb layout screenshot.png
```

Returns JSON with detected panels, toolbars, sidebars, terminal panes, etc.

### learn — store a correction

```bash
screenkb learn screenshot.png screen_type=login_form
screenkb learn photo.jpg screen_type=people_photo
screenkb learn scan.png screen_type=document
```

Stores the image's feature fingerprint in the SQLite knowledge base. Future similar images gain a confidence boost toward that `screen_type`.

### explain — show matched rules

```bash
screenkb explain screenshot.png
```

Shows which IF-THEN rules fired, their scores, and the winning classification.

### rules — list all KB rules

```bash
screenkb rules
```

## LLM Enhancement (optional)

Enable with `--llm`. The LLM receives only extracted features + OCR text, never the raw image.

Supported backends:

| Backend | Env var | Model env var |
|---------|---------|---------------|
| DeepSeek | `DEEPSEEK_API_KEY` | `SCREENKB_LLM_MODEL` (default: `deepseek-chat`) |
| OpenAI | `OPENAI_API_KEY` | `SCREENKB_LLM_MODEL` (default: `gpt-4o-mini`) |
| Ollama | `OLLAMA_BASE_URL` (default: `http://localhost:11434`) | `SCREENKB_LLM_MODEL` (default: `llama3`) |

Backend selection: `--llm-backend deepseek|openai|ollama` or `SCREENKB_LLM` env var or auto-detect from API keys.

## Knowledge Base

Location: `%APPDATA%\screenkb\screenkb.db` (Windows) / `~/.config/screenkb/screenkb.db` (Linux)

Override: `SCREENKB_DB=/path/to/screenkb.db`

Initialize schema manually:
```bash
python migrate_db.py
```

## Compile to .exe (optional)

```bash
pip install pyinstaller
pyinstaller --onefile --name screenkb screenkb/cli.py
# Output: dist/screenkb.exe
# Copy to: tools/bin/screenkb.exe
```
