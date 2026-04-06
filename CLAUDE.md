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
- **Bold detection**: Google Document AI `style_info.bold` / `font_weight ≥ 700` per token, with proportional mapping via `styleSpans` for per-word bold in mixed regions. Falls back to K-means text pixel stroke ratio > 0.35 when Document AI is unavailable.
- **Per-word styling (styleSpans)**: Document AI returns per-token bold/italic/color/fontFamily. `match_region_style_spans()` builds proportional [0.0–1.0] spans, `map_styles_to_words()` maps them to translated text words. Both Fabric.js editor and server-side Pillow renderer apply per-word styles.
- **Fabric.js per-character styles**: `styleSpans` are converted to Fabric.js `styles` object (keyed by unwrapped line index) for per-character bold/italic/color in the editor. Devanagari combining marks inherit the style of their preceding base character to avoid broken ligatures.
- **Color noise filtering**: Document AI per-token colors are noisy (affected by background/anti-aliasing). Per-word color is only applied when spans have truly distinct colors (RGB Manhattan distance > 150), otherwise the single K-means region color is used.
- **Non-Latin font safety**: `get_font()` forces `sans-serif` for non-Latin-script languages (Hindi, Bengali, etc.) because `NotoSerif-Regular.ttf` lacks Devanagari glyphs — prevents tofu (□□□□) rendering.
- **Alignment detection**: Heuristic based on bounding box x-position relative to image width
- **Font sizing**: Binary search for largest font size where wrapped text fits bounding box
- **Font size normalization**: Spatially close regions with similar original font sizes (within 30% tolerance, 120px proximity) are grouped and normalized to median size for visual consistency (e.g., bullet list items).
- **Variable fonts**: Noto Sans supports weight axis — bold via `font.set_variation_by_name("Bold")`
- **Base64 data URLs**: Images passed between frontend and backend as `data:image/png;base64,...`

## Document AI Integration
- **Env var**: `GOOGLE_DOCAI_PROCESSOR=projects/PROJECT_ID/locations/REGION/processors/PROCESSOR_ID`
- **Config**: Set in `.env` file, loaded via `python-dotenv`
- **API**: Uses `documentai_v1.DocumentProcessorServiceClient` with `OcrConfig.PremiumFeatures(compute_style_info=True)`
- **Fallback**: When Document AI is not configured, falls back to pixel-based K-means bold/color detection
- **Token data used**: `style_info.bold`, `style_info.font_weight`, `style_info.pixel_font_size`, `style_info.text_color`, `style_info.italic`, `style_info.font_type`

## Sensitive Files
- `my-project-20223-*.json` — Google Cloud service account key (not currently used, switched to ADC)
- `.env` — Contains `GOOGLE_DOCAI_PROCESSOR` resource name
- Do not commit credentials or `.env` files
