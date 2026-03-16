# Image Text Translator

## Overview
Web app that translates text in images while preserving the original visual layout. Upload an image with text in one language → get it back with translated text that matches the original styling.

## Tech Stack
- **Backend**: FastAPI (Python), served with uvicorn on port 5000
- **OCR**: Google Cloud Vision API (paragraph-level `full_text_annotation`)
- **Translation**: deep-translator (GoogleTranslator)
- **Inpainting**: simple-lama-inpainting (LaMa model — loaded once at startup, heavy)
- **Text rendering**: Pillow + Google Noto variable fonts (support bold via `set_variation_by_name`)
- **Color detection**: scikit-learn KMeans clustering (text vs background pixel separation)
- **Frontend**: Vanilla JS + Fabric.js 5.3.1 (canvas editor)
- **Auth**: Google Application Default Credentials (ADC) — `gcloud auth application-default login`

## Project Structure
```
app.py              — All backend logic: endpoints, OCR, translation, inpainting, rendering
templates/index.html — Single-page HTML with 5 views (upload, select, review, editor, export)
static/app.js       — Frontend state machine, Fabric.js canvas, phase transitions
static/style.css    — Catppuccin Mocha dark theme
fonts/              — Noto Sans variable fonts (Regular supports Bold weight axis)
download_fonts.py   — Downloads required fonts from Google Fonts GitHub
requirements.txt    — Python dependencies
```

## 4-Phase Pipeline
1. **Select Text** — OCR detects text regions → user selects which to clean (click to toggle, draw custom boxes)
2. **Review Clean** — Only selected regions get inpainted via LaMa → before/after comparison
3. **Edit Text** — Fabric.js editor with translated text layers (move, resize, restyle)
4. **Export** — Server-side Pillow rendering → download full-resolution PNG

## Key Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/phase1-detect` | OCR + translate, returns regions (no inpainting) |
| POST | `/phase1-clean` | Inpaints user-selected regions, returns clean image + layer metadata |
| POST | `/phase2-render` | Server-side text rendering from layer JSON |

## Running
```bash
pip install -r requirements.txt
python download_fonts.py
gcloud auth application-default login
python app.py
# → http://127.0.0.1:5000
```

## Key Implementation Details
- **OCR filtering**: Skips regions with height < 14px, width < 14px, text < 2 chars, and small square regions with short text (icon detection)
- **Adaptive mask dilation**: Per-region `dilation = max(3, box_h // 6)` using OpenCV elliptical kernel
- **Bold detection**: K-means text pixel stroke ratio > 0.35 → bold
- **Alignment detection**: Heuristic based on bounding box x-position relative to image width
- **Font sizing**: Binary search for largest font size where wrapped text fits bounding box
- **Variable fonts**: Noto Sans supports weight axis — bold via `font.set_variation_by_name("Bold")`
- **Base64 data URLs**: Images passed between frontend and backend as `data:image/png;base64,...`

## Sensitive Files
- `my-project-20223-*.json` — Google Cloud service account key (not currently used, switched to ADC)
- Do not commit credentials or `.env` files
