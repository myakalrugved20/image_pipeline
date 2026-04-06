import os
import sys
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import asyncio
import io
import json
import base64
import textwrap
from pathlib import Path

# Uses Application Default Credentials (ADC) — run:
#   gcloud auth application-default login
# to authenticate with your Google account.

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import stats
from fastapi import FastAPI, UploadFile, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google.cloud import vision
from deep_translator import GoogleTranslator
from simple_lama_inpainting import SimpleLama
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

# Support Google credentials from environment (for Docker/HF Spaces)
_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if _creds_json and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    _creds_path = Path("/tmp/gcloud_creds.json")
    _creds_path.write_text(_creds_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_creds_path)

BASE_DIR = Path(__file__).resolve().parent
FONT_DIR = BASE_DIR / "fonts"

app = FastAPI(title="Image Text Translator")
app.state.max_request_size = 50 * 1024 * 1024  # 50MB

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Lazy-load LaMa (heavy model) — deferred so the server starts fast on HF Spaces
import torch
_lama = None

def get_lama():
    global _lama
    if _lama is None:
        print("Loading LaMa inpainting model …")
        if not torch.cuda.is_available():
            _orig_jit_load = torch.jit.load
            torch.jit.load = lambda f, **kw: _orig_jit_load(f, **{**kw, "map_location": "cpu"})
            _lama = SimpleLama()
            torch.jit.load = _orig_jit_load
        else:
            _lama = SimpleLama()
        print("LaMa ready.")
    return _lama

# Google Vision client
vision_client = vision.ImageAnnotatorClient()

# Google Document AI client (optional — for font style detection)
_docai_client = None
_docai_processor = os.environ.get("GOOGLE_DOCAI_PROCESSOR", "")  # e.g. "projects/PROJECT/locations/REGION/processors/PROC_ID"

def get_docai_client():
    global _docai_client
    if _docai_client is not None:
        return _docai_client
    if _docai_processor:
        try:
            from google.cloud import documentai_v1 as documentai
            _docai_client = documentai.DocumentProcessorServiceClient()
            print("[DocAI] Client initialized successfully")
            return _docai_client
        except Exception as e:
            print(f"[DocAI] Failed to initialize: {e}")
            return None
    return None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LANG_MAP = None  # lazy-loaded


def _get_languages() -> dict:
    """Return {code: name} dict of supported languages."""
    global LANG_MAP
    if LANG_MAP is None:
        LANG_MAP = GoogleTranslator(source="auto", target="en").get_supported_languages(
            as_dict=True
        )
    return LANG_MAP


# ── OCR ───────────────────────────────────────────────────────────────────

def detect_text(image_bytes: bytes) -> list[dict]:
    """Use Google Vision full_text_annotation to get paragraph-level blocks.

    Filters out small/icon text regions to avoid damaging graphics.
    """
    image = vision.Image(content=image_bytes)
    try:
        response = vision_client.document_text_detection(image=image)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vision API error: {exc}")

    if response.error.message:
        raise HTTPException(status_code=502, detail=f"Vision API error: {response.error.message}")

    if not response.full_text_annotation.pages:
        return []

    regions = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                text = ""
                word_heights = []
                for word in paragraph.words:
                    word_text = "".join(s.text for s in word.symbols)
                    text += word_text + " "
                    # Measure word bounding box height for font size estimation
                    wv = word.bounding_box.vertices
                    if wv:
                        wys = [v.y for v in wv]
                        wh = max(wys) - min(wys)
                        if wh > 0:
                            word_heights.append(wh)
                text = text.strip()
                if not text:
                    continue

                verts = paragraph.bounding_box.vertices
                vertices = [(v.x, v.y) for v in verts]

                # ── Filter out small / icon text ──
                xs = [v[0] for v in vertices]
                ys = [v[1] for v in vertices]
                box_w = max(xs) - min(xs)
                box_h = max(ys) - min(ys)

                area = box_w * box_h
                aspect = box_w / box_h if box_h > 0 else 999
                word_count = len(text.split())

                if box_h < 4 or box_w < 4:
                    print(f"[OCR SKIP] size filter: {box_w}x{box_h} — '{text[:50]}'")
                    continue
                if len(text.strip()) < 2:
                    print(f"[OCR SKIP] short text: '{text}' — {box_w}x{box_h}")
                    continue

                # Icon detection: small, roughly-square region with short text
                if area < 10000 and word_count <= 2 and 0.7 < aspect < 1.4:
                    print(f"[OCR SKIP] icon filter: {box_w}x{box_h} area={area} aspect={aspect:.2f} words={word_count} — '{text[:50]}'")
                    continue

                # Estimate font size from median word height
                median_word_h = 0
                if word_heights:
                    sorted_wh = sorted(word_heights)
                    mid = len(sorted_wh) // 2
                    median_word_h = sorted_wh[mid] if len(sorted_wh) % 2 else (sorted_wh[mid - 1] + sorted_wh[mid]) // 2

                print(f"[OCR KEEP] {box_w}x{box_h} area={area} wordH={median_word_h} — '{text[:60]}'")
                regions.append({"text": text, "vertices": vertices, "word_height": median_word_h})

    return regions


# ── Translation ───────────────────────────────────────────────────────────

def translate_texts(texts: list[str], source: str, target: str) -> list[str]:
    """Translate a list of strings via Google Translate."""
    if not texts:
        return []
    try:
        translator = GoogleTranslator(source=source, target=target)
        results = []
        for t in texts:
            try:
                r = translator.translate(t)
                # Detect error responses returned as "translations"
                if r and ("Error" in r and "Server Error" in r):
                    print(f"[TRANSLATE] Error response detected, keeping original: '{t[:40]}'")
                    results.append(t)
                else:
                    results.append(r if r else t)
            except Exception:
                print(f"[TRANSLATE] Failed for: '{t[:40]}', keeping original")
                results.append(t)
        return results
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Translation error: {exc}")


# ── Mask creation (adaptive dilation) ─────────────────────────────────────

def _dilate_mask(arr: np.ndarray, dilation: int) -> np.ndarray:
    """Dilate a binary mask using OpenCV's fast circular kernel."""
    if dilation <= 0:
        return arr
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
    return cv2.dilate(arr, kernel, iterations=1)


def _text_pixel_mask(image_arr: np.ndarray, verts: list[tuple[int, int]],
                     image_size: tuple[int, int]) -> np.ndarray:
    """Create a mask of likely text pixels within a polygon region.

    Uses edge detection + adaptive thresholding inside the bounding box
    to find text strokes rather than filling the entire polygon.
    Falls back to polygon fill for very small regions.
    """
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    x0, y0 = max(0, min(xs)), max(0, min(ys))
    x1, y1 = min(image_size[0], max(xs)), min(image_size[1], max(ys))
    box_w, box_h = x1 - x0, y1 - y0

    # For very small regions, fall back to polygon fill
    if box_w < 20 or box_h < 12:
        full = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        poly = Image.new("L", image_size, 0)
        ImageDraw.Draw(poly).polygon(verts, fill=255)
        return np.array(poly)

    # Crop the image region
    crop = image_arr[y0:y1, x0:x1]
    if crop.size == 0:
        full = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        poly = Image.new("L", image_size, 0)
        ImageDraw.Draw(poly).polygon(verts, fill=255)
        return np.array(poly)

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop

    # Adaptive threshold to find text pixels (works on varied backgrounds)
    block_size = max(11, (min(box_w, box_h) // 4) | 1)  # ensure odd
    text_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 8
    )

    # Also use Canny edges to catch text strokes
    edges = cv2.Canny(gray, 50, 150)

    # Combine: text pixels from threshold OR edges
    combined = np.maximum(text_thresh, edges)

    # Clip to the polygon region (don't mask outside the OCR polygon)
    poly_mask_full = Image.new("L", image_size, 0)
    ImageDraw.Draw(poly_mask_full).polygon(verts, fill=255)
    poly_arr = np.array(poly_mask_full)
    poly_crop = poly_arr[y0:y1, x0:x1]
    combined = cv2.bitwise_and(combined, poly_crop)

    # Place back into full-size mask
    full = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    full[y0:y1, x0:x1] = combined
    return full


def create_mask(image_size: tuple[int, int], regions: list[dict],
                image: Image.Image = None) -> Image.Image:
    """Create a binary mask with per-region adaptive dilation.

    If `image` is provided, uses text-pixel detection for precise masking.
    Otherwise falls back to polygon fill.
    """
    final_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    image_arr = np.array(image.convert("RGB")) if image is not None else None

    for r in regions:
        verts = r["vertices"]
        ys = [v[1] for v in verts]
        box_h = max(ys) - min(ys)
        dilation = max(5, min(box_h // 5, 25))

        if image_arr is not None:
            # Text-pixel-based mask — only mask actual text strokes
            temp_arr = _text_pixel_mask(image_arr, verts, image_size)
        else:
            # Fallback: solid polygon fill
            temp = Image.new("L", image_size, 0)
            ImageDraw.Draw(temp).polygon(verts, fill=255)
            temp_arr = np.array(temp)

        # Dilate with region-specific radius
        temp_arr = _dilate_mask(temp_arr, dilation)

        # OR into final mask
        final_mask = np.maximum(final_mask, temp_arr)

    return Image.fromarray(final_mask)


# ── Inpainting ────────────────────────────────────────────────────────────

def inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Remove text using LaMa."""
    result = get_lama()(image.convert("RGB"), mask.convert("L"))
    return result


def _region_area(region: dict) -> int:
    """Compute bounding-box area from polygon vertices."""
    verts = region["vertices"]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def inpaint_sequential(image: Image.Image, regions: list[dict],
                       mask_mode: str = "precise") -> Image.Image:
    """Inpaint regions one at a time, smallest first, for better quality.

    mask_mode: "precise" uses text-pixel detection, "fill" uses polygon fill.
    """
    if not regions:
        return image
    use_image = image if mask_mode == "precise" else None
    # Fallback to single pass when there are many regions (performance)
    if len(regions) > 15:
        mask = create_mask(image.size, regions, image=use_image)
        return inpaint(image, mask)
    sorted_regions = sorted(regions, key=_region_area)
    current = image
    for r in sorted_regions:
        mask = create_mask(current.size, [r], image=current if mask_mode == "precise" else None)
        current = inpaint(current, mask)
    return current


# ── Text colour sampling (K-means) ───────────────────────────────────────

def sample_text_color(
    original: Image.Image,
    inpainted: Image.Image,
    vertices: list[tuple[int, int]],
) -> tuple[tuple[int, int, int], bool]:
    """Estimate text colour and boldness using K-means clustering.

    Returns (rgb_color, is_bold).
    """
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(original.width, x1), min(original.height, y1)

    orig_crop = np.array(original.crop((x0, y0, x1, y1))).astype(float)
    inp_crop = np.array(inpainted.crop((x0, y0, x1, y1))).astype(float)

    if orig_crop.size == 0 or orig_crop.shape[0] < 2 or orig_crop.shape[1] < 2:
        return (0, 0, 0), False

    # Per-pixel diff magnitude
    diff = np.sqrt(np.sum((orig_crop - inp_crop) ** 2, axis=2))
    flat_diff = diff.flatten()

    if flat_diff.shape[0] < 4:
        return (0, 0, 0), False

    try:
        # K-means with k=2 to separate text pixels (high diff) from background (low diff)
        km = KMeans(n_clusters=2, n_init=3, random_state=0)
        labels = km.fit_predict(flat_diff.reshape(-1, 1))

        # The cluster with the higher centroid is the text cluster
        if km.cluster_centers_[0, 0] > km.cluster_centers_[1, 0]:
            text_label = 0
        else:
            text_label = 1

        # Reshape labels to 2D
        label_map = labels.reshape(diff.shape)
        text_mask = label_map == text_label

        # Only use pixels with meaningful diff (skip near-zero diffs)
        min_diff = max(km.cluster_centers_[:, 0]) * 0.3
        text_mask = text_mask & (diff > min_diff)

        if not np.any(text_mask):
            raise ValueError("No text pixels found")

        text_pixels = orig_crop[text_mask]
        median_color = np.median(text_pixels, axis=0).astype(int)

        # Bold detection: if text pixels occupy a large fraction → bold stroke
        text_pixel_count = int(np.sum(text_mask))
        total_pixels = text_mask.size
        stroke_ratio = text_pixel_count / total_pixels if total_pixels > 0 else 0
        is_bold = stroke_ratio > 0.35

        return tuple(median_color.tolist()), is_bold

    except Exception:
        # Fallback: old percentile method
        threshold = np.percentile(flat_diff, 70)
        text_mask = diff >= threshold
        if not np.any(text_mask):
            return (0, 0, 0), False
        text_pixels = orig_crop[text_mask]
        median_color = np.median(text_pixels, axis=0).astype(int)
        return tuple(median_color.tolist()), False


# ── Style detection (italic, underline, font family) ─────────────────────

def detect_italic(
    original: Image.Image,
    inpainted: Image.Image,
    vertices: list[tuple[int, int]],
) -> bool:
    """Detect italic text by measuring the slant of text pixel centroids."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(original.width, x1), min(original.height, y1)

    orig_crop = np.array(original.crop((x0, y0, x1, y1))).astype(float)
    inp_crop = np.array(inpainted.crop((x0, y0, x1, y1))).astype(float)

    if orig_crop.size == 0 or orig_crop.shape[0] < 4 or orig_crop.shape[1] < 4:
        return False

    diff = np.sqrt(np.sum((orig_crop - inp_crop) ** 2, axis=2))
    threshold = np.percentile(diff.flatten(), 70)
    text_mask = diff >= threshold

    # For each row, compute the centroid (avg x) of text pixels
    row_indices = []
    centroids = []
    for row_i in range(text_mask.shape[0]):
        cols = np.where(text_mask[row_i])[0]
        if len(cols) >= 3:
            row_indices.append(row_i)
            centroids.append(np.mean(cols))

    if len(row_indices) < 5:
        return False

    # Linear regression: if slope is significant, text is slanted (italic)
    slope, _, r_value, _, _ = stats.linregress(row_indices, centroids)
    # Very strict thresholds — pixel-based italic detection is unreliable on
    # gradient/textured backgrounds, so only flag obvious cases
    return abs(slope) > 0.4 and abs(r_value) > 0.85


def detect_font_family(
    original: Image.Image,
    vertices: list[tuple[int, int]],
) -> str:
    """Detect serif vs sans-serif using edge frequency analysis."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(original.width, x1), min(original.height, y1)

    crop = np.array(original.crop((x0, y0, x1, y1)).convert("L"))
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return "sans-serif"

    # Binarize to find text pixels
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_pixel_count = np.sum(binary > 0)
    if text_pixel_count < 20:
        return "sans-serif"

    # Horizontal Sobel → detects vertical edges (serifs create many fine edges)
    sobel_h = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
    # Only count edges on text pixels
    edge_on_text = np.abs(sobel_h) * (binary > 0)
    edge_count = np.sum(edge_on_text > 30)

    ratio = edge_count / text_pixel_count if text_pixel_count > 0 else 0
    return "serif" if ratio > 0.45 else "sans-serif"


def detect_underline(
    original: Image.Image,
    inpainted: Image.Image,
    vertices: list[tuple[int, int]],
) -> bool:
    """Detect underline by looking for a horizontal line in the bottom portion."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(original.width, x1), min(original.height, y1)
    box_w = x1 - x0
    box_h = y1 - y0

    if box_w < 10 or box_h < 10:
        return False

    orig_crop = np.array(original.crop((x0, y0, x1, y1))).astype(float)
    inp_crop = np.array(inpainted.crop((x0, y0, x1, y1))).astype(float)

    diff = np.sqrt(np.sum((orig_crop - inp_crop) ** 2, axis=2))

    # Look at bottom 25% of the box
    bottom_start = int(box_h * 0.75)
    bottom_region = diff[bottom_start:, :]

    if bottom_region.size == 0:
        return False

    threshold = np.percentile(diff.flatten(), 60)

    # Check each row for a long horizontal run of changed pixels
    # Use strict threshold — on gradient backgrounds, the inpainting diff
    # in the bottom region can falsely trigger underline detection
    strong_threshold = np.percentile(diff.flatten(), 80)
    for row in bottom_region:
        run = np.sum(row >= strong_threshold)
        if run > box_w * 0.75:
            return True

    return False


def detect_font_styles_docai(image_bytes: bytes) -> dict | None:
    """Call Google Document AI to get font style metadata.

    Returns dict with 'full_text' and 'styles' list, or None if not configured.
    The 'styles' list uses the same format as the old Azure DI path so that
    match_region_style() works without changes.
    """
    client = get_docai_client()
    if not client:
        return None

    try:
        from google.cloud import documentai_v1 as documentai

        raw_document = documentai.RawDocument(
            content=image_bytes,
            mime_type="image/png",
        )
        request = documentai.ProcessRequest(
            name=_docai_processor,
            raw_document=raw_document,
            process_options=documentai.ProcessOptions(
                ocr_config=documentai.OcrConfig(
                    premium_features=documentai.OcrConfig.PremiumFeatures(
                        compute_style_info=True,
                    ),
                ),
            ),
        )
        result = client.process_document(request=request)
        document = result.document
        full_text = document.text or ""

        # Build style entries from token-level style_info
        styles = []
        for page in document.pages:
            for token in page.tokens:
                si = token.style_info
                if not si:
                    continue
                # Get token text from text_anchor
                token_text = ""
                offset = 0
                length = 0
                if token.layout and token.layout.text_anchor and token.layout.text_anchor.text_segments:
                    seg = token.layout.text_anchor.text_segments[0]
                    offset = seg.start_index
                    length = seg.end_index - seg.start_index
                    token_text = full_text[offset:seg.end_index]

                style_entry = {
                    "offset": offset,
                    "length": length,
                    "text": token_text,
                }
                # Bold — direct bool or weight >= 700
                if si.bold:
                    style_entry["fontWeight"] = "bold"
                elif si.font_weight and si.font_weight >= 700:
                    style_entry["fontWeight"] = "bold"
                else:
                    style_entry["fontWeight"] = "normal"
                # Italic
                if si.italic:
                    style_entry["fontStyle"] = "italic"
                # Font size — use pixel_font_size directly (already in pixels)
                if si.pixel_font_size and si.pixel_font_size > 0:
                    style_entry["fontSize"] = si.pixel_font_size
                elif si.font_size and si.font_size > 0:
                    style_entry["fontSize"] = si.font_size * 1.333  # pt to px
                # Underline
                if getattr(si, "underlined", False):
                    style_entry["textDecoration"] = "underline"
                # Font family from font_type (SANS_SERIF, SERIF, etc.)
                ft = getattr(si, "font_type", "") or ""
                if "SERIF" in ft and "SANS" not in ft:
                    style_entry["fontFamily"] = "serif"
                elif ft:
                    style_entry["fontFamily"] = "sans-serif"
                # Color — from style_info.text_color (0.0–1.0 floats)
                if si.text_color:
                    r = int((si.text_color.red or 0) * 255)
                    g = int((si.text_color.green or 0) * 255)
                    b = int((si.text_color.blue or 0) * 255)
                    style_entry["color"] = f"#{r:02x}{g:02x}{b:02x}"

                styles.append(style_entry)

        print(f"[DocAI] Detected {len(styles)} style tokens in {len(full_text)} chars")
        for st in styles:
            print(f"  → '{st.get('text', '')[:30]}' weight={st.get('fontWeight')} color={st.get('color')} size={st.get('fontSize')}")

        return {"full_text": full_text, "styles": styles}

    except Exception as e:
        print(f"[DocAI] Error: {e}")
        return None


def _find_text_fuzzy(full_text: str, search: str) -> int:
    """Find substring in full_text with fuzzy matching.

    Tries: exact → case-insensitive → newline-normalized + case-insensitive.
    Returns the index in the ORIGINAL full_text.
    """
    # Exact match
    idx = full_text.find(search)
    if idx != -1:
        return idx
    # Case-insensitive
    idx = full_text.lower().find(search.lower())
    if idx != -1:
        return idx
    # Normalize newlines to spaces and try case-insensitive
    normalized = full_text.replace("\n", " ").replace("  ", " ")
    search_norm = search.replace("\n", " ").replace("  ", " ")
    norm_idx = normalized.lower().find(search_norm.lower())
    if norm_idx != -1:
        # Map normalized index back to original full_text index
        # Count how many chars in original up to the same logical position
        orig_idx = 0
        norm_pos = 0
        while norm_pos < norm_idx and orig_idx < len(full_text):
            if full_text[orig_idx] == "\n":
                # newline became space in normalized — still counts as 1 char
                pass
            orig_idx += 1
            norm_pos += 1
        return orig_idx
    return -1


def match_region_style(region_text: str, docai_data: dict | None) -> dict | None:
    """Match an OCR region's text to Document AI style data.

    Returns style dict or None if no match found.
    """
    if not docai_data or not docai_data.get("styles"):
        return None

    full_text = docai_data["full_text"]
    styles = docai_data["styles"]

    # Try to find the region text (fuzzy: ignore whitespace and case differences)
    region_clean = region_text.strip().replace("  ", " ")

    # Collect all styles that overlap with this region's text
    matching_styles = []

    # Try substring match (exact then case-insensitive)
    idx = _find_text_fuzzy(full_text, region_clean)
    if idx == -1:
        # Try matching first few words
        words = region_clean.split()
        if len(words) >= 2:
            prefix = " ".join(words[:3])
            idx = _find_text_fuzzy(full_text, prefix)

    if idx == -1:
        return None

    region_start = idx
    region_end = idx + len(region_clean)

    for s in styles:
        s_start = s["offset"]
        s_end = s_start + s["length"]
        # Check overlap
        if s_start < region_end and s_end > region_start:
            matching_styles.append(s)

    if not matching_styles:
        return None

    # Weighted voting across all overlapping tokens — each token's vote
    # is weighted by its character length so majority style wins
    total_chars = sum(s["length"] for s in matching_styles)
    if total_chars == 0:
        return None

    bold_chars = sum(s["length"] for s in matching_styles if s.get("fontWeight", "").lower() == "bold")
    italic_chars = sum(s["length"] for s in matching_styles if s.get("fontStyle", "").lower() == "italic")
    underline_chars = sum(s["length"] for s in matching_styles if "underline" in s.get("textDecoration", "").lower())

    result = {}
    result["is_bold"] = bold_chars > total_chars * 0.5
    result["is_italic"] = italic_chars > total_chars * 0.5
    result["is_underline"] = underline_chars > total_chars * 0.5

    # Font family — majority vote
    family_counts = {}
    for s in matching_styles:
        ff = s.get("fontFamily") or s.get("similarFontFamily") or "sans-serif"
        family_counts[ff] = family_counts.get(ff, 0) + s["length"]
    result["font_family"] = max(family_counts, key=family_counts.get)

    # Font size — weighted average across tokens
    size_sum = 0
    size_weight = 0
    for s in matching_styles:
        fs = s.get("fontSize")
        if fs and fs > 0:
            size_sum += float(fs) * s["length"]
            size_weight += s["length"]
    if size_weight > 0:
        result["font_size"] = round(size_sum / size_weight)

    # Color — use the most common color (by character coverage)
    color_counts = {}
    for s in matching_styles:
        c = s.get("color")
        if c and c.startswith("#") and len(c) == 7:
            color_counts[c] = color_counts.get(c, 0) + s["length"]
    if color_counts:
        best_color = max(color_counts, key=color_counts.get)
        r = int(best_color[1:3], 16)
        g = int(best_color[3:5], 16)
        b = int(best_color[5:7], 16)
        result["color"] = (r, g, b)

    print(f"[DocAI] Matched '{region_text[:40]}' → bold={result.get('is_bold')} italic={result.get('is_italic')} family={result.get('font_family')} size={result.get('font_size')} (bold_ratio={bold_chars}/{total_chars})")
    return result


def match_region_style_spans(region_text: str, docai_data: dict | None) -> list[dict] | None:
    """Return proportional style spans for a region's text.

    Each span covers a [start, end) range normalized to 0.0–1.0 of the region
    text length, with per-span fontWeight, fontStyle, color, fontFamily.
    Returns None if no match found.
    """
    if not docai_data or not docai_data.get("styles"):
        return None

    full_text = docai_data["full_text"]
    styles = docai_data["styles"]

    region_clean = region_text.strip().replace("  ", " ")
    idx = _find_text_fuzzy(full_text, region_clean)
    if idx == -1:
        words = region_clean.split()
        if len(words) >= 2:
            prefix = " ".join(words[:3])
            idx = _find_text_fuzzy(full_text, prefix)
    if idx == -1:
        return None

    region_start = idx
    region_end = idx + len(region_clean)
    region_len = max(1, len(region_clean))

    # Build per-character style array from overlapping tokens
    # Default style for characters not covered by any token
    default_style = {"fontWeight": "normal", "fontStyle": "normal",
                     "color": None, "fontFamily": "sans-serif"}
    char_styles = [dict(default_style) for _ in range(region_len)]

    for s in styles:
        s_start = s["offset"]
        s_end = s_start + s["length"]
        # Clamp to region bounds
        ov_start = max(s_start, region_start) - region_start
        ov_end = min(s_end, region_end) - region_start
        if ov_start >= ov_end:
            continue
        for ci in range(ov_start, ov_end):
            cs = char_styles[ci]
            if s.get("fontWeight"):
                cs["fontWeight"] = s["fontWeight"]
            if s.get("fontStyle"):
                cs["fontStyle"] = s["fontStyle"]
            if s.get("color"):
                cs["color"] = s["color"]
            if s.get("fontFamily"):
                cs["fontFamily"] = s["fontFamily"]

    # Merge consecutive characters with identical style into spans
    spans = []
    i = 0
    while i < region_len:
        cur = char_styles[i]
        j = i + 1
        while j < region_len and char_styles[j] == cur:
            j += 1
        spans.append({
            "start": i / region_len,
            "end": j / region_len,
            "fontWeight": cur["fontWeight"],
            "fontStyle": cur["fontStyle"],
            "color": cur["color"],
            "fontFamily": cur["fontFamily"],
        })
        i = j

    return spans


def map_styles_to_words(translated_text: str, style_spans: list[dict]) -> list[dict]:
    """Map proportional style spans to individual words of translated text.

    Returns list of {"word": str, "style": dict} entries.
    Each word gets the style of the span covering its midpoint.
    """
    words = translated_text.split()
    if not words or not style_spans:
        return [{"word": w, "style": style_spans[0] if style_spans else {}} for w in words]

    total_len = len(translated_text)
    if total_len == 0:
        return [{"word": w, "style": style_spans[0]} for w in words]

    result = []
    pos = 0
    for word in words:
        word_start = translated_text.find(word, pos)
        if word_start == -1:
            word_start = pos
        word_mid = (word_start + word_start + len(word)) / 2.0
        prop = word_mid / total_len

        # Find the span covering this proportional position
        matched_style = style_spans[-1]  # default to last span
        for span in style_spans:
            if span["start"] <= prop < span["end"]:
                matched_style = span
                break

        result.append({"word": word, "style": matched_style})
        pos = word_start + len(word)

    return result


def analyze_text_style(
    original: Image.Image,
    inpainted: Image.Image,
    vertices: list[tuple[int, int]],
    docai_styles: dict | None = None,
    region_text: str = "",
) -> dict:
    """Comprehensive text style analysis.

    If docai_styles is provided, uses Document AI data for bold/italic/fontFamily.
    Falls back to pixel-based detection otherwise.
    Always uses pixel-based color detection (more reliable for the actual rendered color).
    """
    # Always get color from pixel analysis (Document AI color may not match rendered color)
    color, pixel_bold = sample_text_color(original, inpainted, vertices)

    # Try Document AI first
    docai_match = match_region_style(region_text, docai_styles) if docai_styles and region_text else None

    if docai_match:
        is_bold = docai_match.get("is_bold", pixel_bold)
        is_italic = docai_match.get("is_italic", False)
        is_underline = docai_match.get("is_underline", False)
        font_family = docai_match.get("font_family", "sans-serif")
        # Use Document AI color if pixel detection returned a fallback
        if "color" in docai_match:
            color = docai_match["color"]
    else:
        # Fallback to pixel-based detection
        is_bold = pixel_bold
        is_italic = detect_italic(original, inpainted, vertices)
        is_underline = detect_underline(original, inpainted, vertices)
        font_family = detect_font_family(original, vertices)

    return {
        "color": color,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "is_underline": is_underline,
        "font_family": font_family,
        "_docai_font_size": docai_match.get("font_size") if docai_match else None,  # informational only
    }


# ── Font selection ────────────────────────────────────────────────────────

CJK_LANGS = {"zh-cn", "zh-tw", "ja", "ko", "zh-CN", "zh-TW"}
ARABIC_LANGS = {"ar", "fa", "ur"}

# Map CSS font-family values to Noto font files
FONT_FAMILY_MAP = {
    "sans-serif": "NotoSans-Regular.ttf",
    "Arial":      "NotoSans-Regular.ttf",
    "Verdana":    "NotoSans-Regular.ttf",
    "serif":      "NotoSerif-Regular.ttf",
    "Georgia":    "NotoSerif-Regular.ttf",
    "monospace":  "NotoSansMono-Regular.ttf",
}

ITALIC_FONT_MAP = {
    "sans-serif": "NotoSans-Italic.ttf",
    "Arial":      "NotoSans-Italic.ttf",
    "Verdana":    "NotoSans-Italic.ttf",
    # Serif/mono italic not available — fall back to regular
}


def get_font(
    target_lang: str,
    size: int,
    bold: bool = False,
    italic: bool = False,
    font_family: str = "sans-serif",
) -> ImageFont.FreeTypeFont:
    """Pick the right Noto font for the target language and style."""
    # Non-Latin scripts: NotoSerif doesn't have Devanagari/etc glyphs,
    # so force sans-serif for any language that isn't Latin-script based
    LATIN_LANGS = {"en", "es", "fr", "de", "it", "pt", "nl", "pl", "ro", "cs",
                   "sk", "hr", "sv", "da", "no", "fi", "hu", "tr", "vi", "id",
                   "ms", "tl", "sw", "af", "ca", "eu", "gl", "lt", "lv", "et",
                   "sl", "mt", "sq", "cy", "ga", "is"}
    if target_lang not in LATIN_LANGS and font_family != "sans-serif":
        font_family = "sans-serif"

    if target_lang in CJK_LANGS:
        candidates = ["NotoSansCJKsc-Regular.otf", "NotoSansCJK-Regular.ttc"]
    elif target_lang in ARABIC_LANGS:
        candidates = ["NotoSansArabic-Regular.ttf"]
    else:
        # Try italic variant first if requested
        if italic and font_family in ITALIC_FONT_MAP:
            italic_name = ITALIC_FONT_MAP[font_family]
            if (FONT_DIR / italic_name).exists():
                candidates = [italic_name]
            else:
                candidates = [FONT_FAMILY_MAP.get(font_family, "NotoSans-Regular.ttf")]
        else:
            candidates = [FONT_FAMILY_MAP.get(font_family, "NotoSans-Regular.ttf")]

    for name in candidates:
        path = FONT_DIR / name
        if path.exists():
            font = ImageFont.truetype(str(path), size)
            if bold:
                try:
                    font.set_variation_by_name("Bold")
                except Exception:
                    pass
            return font

    for fallback in ["arial.ttf", "segoeui.ttf"]:
        try:
            font = ImageFont.truetype(fallback, size)
            return font
        except OSError:
            continue

    return ImageFont.load_default()


# ── Text rendering helpers (pixel-accurate) ──────────────────────────────

def _is_cjk_char(ch: str) -> bool:
    """Check if a character is CJK (Chinese, Japanese, Korean)."""
    cp = ord(ch)
    return any((
        0x4E00 <= cp <= 0x9FFF,    # CJK Unified Ideographs
        0x3400 <= cp <= 0x4DBF,    # CJK Extension A
        0x3000 <= cp <= 0x303F,    # CJK Symbols and Punctuation
        0x3040 <= cp <= 0x309F,    # Hiragana
        0x30A0 <= cp <= 0x30FF,    # Katakana
        0xAC00 <= cp <= 0xD7AF,    # Hangul Syllables
        0xFF00 <= cp <= 0xFFEF,    # Fullwidth Forms
        0x20000 <= cp <= 0x2A6DF,  # CJK Extension B
    ))


def _has_cjk(text: str) -> bool:
    """Check if text contains any CJK characters."""
    return any(_is_cjk_char(ch) for ch in text)


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-based wrapping with CJK character-level breaking."""
    if not text.strip():
        return [text]

    if _has_cjk(text):
        return _wrap_text_cjk(text, font, max_width)

    # Standard space-based wrapping for non-CJK text
    words = text.split()
    if not words:
        return [text]
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        bbox = font.getbbox(test)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [text]


def _wrap_text_cjk(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Character-level wrapping for CJK text. Breaks between any CJK chars."""
    lines = []
    current = ""
    for ch in text:
        test = current + ch
        bbox = font.getbbox(test)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines if lines else [text]


def _wrap_text_styled(
    word_styles: list[dict],
    target_lang: str,
    max_width: int,
    base_font_size: int,
) -> list[list[dict]]:
    """Wrap styled words into lines that fit max_width.

    Each word gets its own font based on its style. Returns list of lines,
    where each line is a list of {"word", "font", "color", "width", "underline"}.
    """
    if not word_styles:
        return []

    # Check if word styles have significantly different colors (not just noise)
    def _parse_hex(h):
        if not h or not h.startswith("#") or len(h) != 7:
            return None
        return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))

    span_colors = [_parse_hex(ws["style"].get("color")) for ws in word_styles]
    span_colors = [c for c in span_colors if c]
    has_distinct_colors = False
    if len(span_colors) >= 2:
        for i in range(len(span_colors)):
            if has_distinct_colors:
                break
            for j in range(i + 1, len(span_colors)):
                dist = (abs(span_colors[i][0] - span_colors[j][0])
                        + abs(span_colors[i][1] - span_colors[j][1])
                        + abs(span_colors[i][2] - span_colors[j][2]))
                if dist > 150:
                    has_distinct_colors = True
                    break

    # Build font and measure width for each word
    entries = []
    for ws in word_styles:
        st = ws["style"]
        is_bold = st.get("fontWeight", "normal").lower() == "bold"
        is_italic = st.get("fontStyle", "normal").lower() == "italic"
        ff = st.get("fontFamily", "sans-serif") or "sans-serif"
        font = get_font(target_lang, base_font_size, bold=is_bold, italic=is_italic, font_family=ff)
        bbox = font.getbbox(ws["word"])
        w = bbox[2] - bbox[0]
        # Only use per-word color when spans have truly distinct colors
        color_hex = st.get("color") if has_distinct_colors else None
        underline = "underline" in st.get("textDecoration", "").lower() if st.get("textDecoration") else False
        entries.append({
            "word": ws["word"],
            "font": font,
            "color_hex": color_hex,
            "width": w,
            "bold": is_bold,
            "underline": underline,
        })

    # Greedy line wrapping (respects explicit \n in words)
    lines = []
    current_line = []
    current_w = 0
    space_w = entries[0]["font"].getbbox(" ")[2] - entries[0]["font"].getbbox(" ")[0] if entries else 4

    for entry in entries:
        # Handle explicit newlines within a word (e.g. user-edited text)
        if "\n" in entry["word"]:
            parts = entry["word"].split("\n")
            for pi, part in enumerate(parts):
                if part:
                    part_entry = dict(entry)
                    part_entry["word"] = part
                    bbox = part_entry["font"].getbbox(part)
                    part_entry["width"] = bbox[2] - bbox[0]
                    test_w = current_w + (space_w if current_line else 0) + part_entry["width"]
                    if test_w <= max_width or not current_line:
                        current_line.append(part_entry)
                        current_w = test_w
                    else:
                        lines.append(current_line)
                        current_line = [part_entry]
                        current_w = part_entry["width"]
                if pi < len(parts) - 1:
                    # Force line break
                    if current_line:
                        lines.append(current_line)
                    current_line = []
                    current_w = 0
            continue

        test_w = current_w + (space_w if current_line else 0) + entry["width"]
        if test_w <= max_width or not current_line:
            current_line.append(entry)
            current_w = test_w
        else:
            lines.append(current_line)
            current_line = [entry]
            current_w = entry["width"]
    if current_line:
        lines.append(current_line)

    return lines


def _render_styled_lines(
    draw: ImageDraw.Draw,
    lines: list[list[dict]],
    x: int, y: int,
    box_w: int, box_h: int,
    alignment: str,
    default_color: tuple,
    alpha: int = 255,
    base_font_size: int = 16,
):
    """Draw word-by-word with per-word font and color."""
    if not lines:
        return

    # Compute line heights
    line_heights = []
    for line in lines:
        max_lh = max(_line_height(e["font"]) for e in line) if line else 0
        line_heights.append(max_lh)
    total_h = sum(line_heights)

    cy = y + max(0, (box_h - total_h) // 2)

    for line, lh in zip(lines, line_heights):
        # Compute line width
        space_w = line[0]["font"].getbbox(" ")[2] - line[0]["font"].getbbox(" ")[0] if line else 4
        line_w = sum(e["width"] for e in line) + space_w * max(0, len(line) - 1)

        if alignment == "left":
            cx = x
        elif alignment == "right":
            cx = x + box_w - line_w
        else:
            cx = x + max(0, (box_w - line_w) // 2)

        for i, entry in enumerate(line):
            # Resolve color
            c_hex = entry.get("color_hex")
            if c_hex and c_hex.startswith("#") and len(c_hex) == 7:
                cr = int(c_hex[1:3], 16)
                cg = int(c_hex[3:5], 16)
                cb = int(c_hex[5:7], 16)
                fill = (cr, cg, cb, alpha)
            else:
                fill = (*default_color, alpha)

            draw.text((cx, cy), entry["word"], fill=fill, font=entry["font"])

            if entry.get("underline"):
                ul_y = cy + lh - 2
                ul_thick = max(1, base_font_size // 14)
                draw.line([(cx, ul_y), (cx + entry["width"], ul_y)],
                          fill=fill, width=ul_thick)

            cx += entry["width"]
            if i < len(line) - 1:
                cx += space_w

        cy += lh


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    """Compute line height from font metrics."""
    bbox = font.getbbox("Ayg|")
    return (bbox[3] - bbox[1]) + 2


def _estimate_original_font_size(orig_text: str, box_h: int, word_height: int = 0,
                                  docai_font_size: int | None = None) -> int:
    """Estimate the original font size from the best available source.

    Priority:
    1. Document AI pixel_font_size (weighted average across matched tokens)
    2. word_height from OCR (median word bounding box height)
    3. box_h / line_count heuristic
    """
    if docai_font_size and docai_font_size > 0:
        return max(6, int(docai_font_size))
    if word_height > 0:
        # Word bbox height ≈ font size (slightly larger due to ascenders/descenders)
        return max(6, int(word_height * 0.85))
    # Fallback: estimate from paragraph box height
    line_count = max(1, orig_text.count("\n") + 1)
    words = orig_text.split()
    if line_count == 1 and len(words) > 6:
        line_count = max(1, len(words) // 4)
    estimated = int(box_h / line_count / 1.2)
    return max(6, estimated)


def _normalize_font_sizes(layers: list[dict], target_lang: str = "", proximity: int = 120, size_tolerance: float = 0.30):
    """Group spatially close regions with similar original font sizes and normalize.

    Regions are grouped if:
    - Their bounding boxes are within `proximity` pixels of each other
    - Their originalFontSize values are within `size_tolerance` ratio of each other

    All regions in a group get the median fontSize in the group,
    ensuring consistent sizing for text from the same visual section.
    """
    if not layers:
        return

    n = len(layers)
    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        # BFS to find all connected regions in this group
        # Check new candidates against the group's min/max size range
        # to prevent BFS chaining (A→B→C where A and C are too different)
        group = [i]
        visited[i] = True
        queue = [i]
        group_min_size = layers[i].get("originalFontSize", 0)
        group_max_size = group_min_size
        while queue:
            cur = queue.pop(0)
            cx = layers[cur]["x"] + layers[cur]["width"] / 2
            cy = layers[cur]["y"] + layers[cur]["height"] / 2
            for j in range(n):
                if visited[j]:
                    continue
                jx = layers[j]["x"] + layers[j]["width"] / 2
                jy = layers[j]["y"] + layers[j]["height"] / 2
                dist = max(abs(cx - jx), abs(cy - jy))  # Chebyshev distance
                j_orig_size = layers[j].get("originalFontSize", 0)
                if dist < proximity and j_orig_size > 0 and group_min_size > 0:
                    # Check against the group's full range, not just the current node
                    new_min = min(group_min_size, j_orig_size)
                    new_max = max(group_max_size, j_orig_size)
                    ratio = new_min / new_max if new_max > 0 else 1
                    if ratio >= (1 - size_tolerance):
                        visited[j] = True
                        group.append(j)
                        queue.append(j)
                        group_min_size = new_min
                        group_max_size = new_max

        if len(group) > 1:
            orig_sizes = [layers[idx].get("originalFontSize", 0) for idx in group]
            # Use median of originalFontSize as the target size for the group
            sorted_orig = sorted(orig_sizes)
            target_orig_size = sorted_orig[len(sorted_orig) // 2]

            # Use majority vote for bold/italic/underline
            bold_votes = sum(1 for idx in group if layers[idx].get("bold"))
            italic_votes = sum(1 for idx in group if layers[idx].get("italic"))
            underline_votes = sum(1 for idx in group if layers[idx].get("underline"))
            majority = len(group) / 2
            group_bold = bold_votes > majority
            group_italic = italic_votes > majority
            group_underline = underline_votes > majority

            # Re-fit each layer's font size using the group's target original size as max
            print(f"[NORMALIZE] group of {len(group)}: origSizes={orig_sizes} → target={target_orig_size}px, bold={group_bold}, italic={group_italic}")
            for idx in group:
                l = layers[idx]
                fit_w = l.get("fitWidth", l["width"])
                fit_h = l.get("fitHeight", l["height"])
                text = l.get("translatedText", "")
                new_size = _fit_font_size(text, target_lang, fit_w, fit_h,
                                          bold=group_bold, max_size=target_orig_size)
                l["fontSize"] = new_size
                l["originalFontSize"] = target_orig_size
                l["bold"] = group_bold
                l["italic"] = group_italic
                l["underline"] = group_underline


def _fit_font_size(text: str, target_lang: str, box_w: int, box_h: int,
                   bold: bool = False, max_size: int = 0) -> int:
    """Binary search for the largest font size where text fits the box.

    If max_size > 0, the result will not exceed max_size (original font size cap).
    """
    upper = max(box_h, 10)
    if max_size > 0:
        upper = min(upper, max_size)
    lo, hi = 6, upper
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        font = get_font(target_lang, mid, bold=bold)
        lines = _wrap_text(text, font, box_w)
        lh = _line_height(font)
        total_h = len(lines) * lh
        max_line_w = max(
            (font.getbbox(line)[2] - font.getbbox(line)[0]) for line in lines
        )
        if max_line_w <= box_w and total_h <= box_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ── Text rendering (with centering) ──────────────────────────────────────

def _calc_rotation(vertices: list) -> float:
    """Calculate rotation angle (degrees) from OCR polygon vertices.

    Vertices order: top-left, top-right, bottom-right, bottom-left.
    Returns angle in degrees (negative = clockwise).
    """
    if not vertices or len(vertices) < 2:
        return 0.0
    v0, v1 = vertices[0], vertices[1]
    dx = v1[0] - v0[0]
    dy = v1[1] - v0[1]
    angle = math.degrees(math.atan2(dy, dx))
    # Snap to 0 if nearly horizontal (within 3 degrees)
    if abs(angle) < 3:
        return 0.0
    return round(angle, 1)


def _rotated_box_dims(vertices: list) -> tuple:
    """Get the actual width/height of the rotated text box from polygon vertices.

    Width = along the text direction (v0→v1 edge).
    Height = perpendicular (v0→v3 edge).
    """
    if not vertices or len(vertices) < 4:
        return (0, 0)
    v0, v1, v2, v3 = vertices[:4]
    w = math.sqrt((v1[0] - v0[0])**2 + (v1[1] - v0[1])**2)
    h = math.sqrt((v3[0] - v0[0])**2 + (v3[1] - v0[1])**2)
    return (round(w), round(h))


def _polygon_center(vertices: list) -> tuple:
    """Get the center point of a polygon."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _detect_alignment(x0: int, box_w: int, img_w: int) -> str:
    """Heuristic alignment detection based on position within the image."""
    center_x = x0 + box_w / 2
    relative = center_x / img_w if img_w > 0 else 0.5
    if relative < 0.4:
        return "left"
    elif relative > 0.6:
        return "right"
    return "center"


def render_text(
    image: Image.Image,
    original: Image.Image,
    inpainted: Image.Image,
    regions: list[dict],
    translated: list[str],
    target_lang: str,
) -> Image.Image:
    """Draw translated text with pixel-accurate sizing, alignment, and bold detection."""
    result = image.copy()
    draw = ImageDraw.Draw(result)

    for region, text in zip(regions, translated):
        verts = region["vertices"]
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)
        box_w = x1 - x0
        box_h = y1 - y0

        if box_w <= 0 or box_h <= 0:
            continue

        color, is_bold = sample_text_color(original, inpainted, verts)
        alignment = _detect_alignment(x0, box_w, image.width)
        font_size = _fit_font_size(text, target_lang, box_w, box_h, bold=is_bold)
        font = get_font(target_lang, font_size, bold=is_bold)
        lines = _wrap_text(text, font, box_w)
        lh = _line_height(font)
        total_h = len(lines) * lh

        # Vertical centering
        start_y = y0 + max(0, (box_h - total_h) // 2)

        for line in lines:
            if start_y + lh > y1 + lh:  # allow slight overflow
                break
            line_bbox = font.getbbox(line)
            line_w = line_bbox[2] - line_bbox[0]
            # Horizontal alignment
            if alignment == "left":
                start_x = x0
            elif alignment == "right":
                start_x = x0 + box_w - line_w
            else:
                start_x = x0 + max(0, (box_w - line_w) // 2)
            draw.text((start_x, start_y), line, fill=color, font=font)
            start_y += lh

    return result


# ── Server-side render from layer JSON ────────────────────────────────────

def render_layers_on_image(
    base_image: Image.Image,
    layers: list[dict],
    target_lang: str,
) -> Image.Image:
    """Render text layers onto an image (server-side, pixel-accurate)."""
    # Work in RGBA for opacity support
    result = base_image.convert("RGBA")

    for layer in layers:
        text = layer.get("text", "")
        if not text:
            continue

        x = int(layer.get("x", 0))
        y = int(layer.get("y", 0))
        w = int(layer.get("width", 200))
        h = int(layer.get("height", 50))
        font_size = int(layer.get("fontSize", 16))
        color_hex = layer.get("color", "#000000")
        opacity = float(layer.get("opacity", 1.0))
        font_weight = layer.get("fontWeight", "normal")
        font_style = layer.get("fontStyle", "normal")
        font_family = layer.get("fontFamily", "sans-serif")
        underline = layer.get("underline", False)
        alignment = layer.get("alignment", "center")
        angle = float(layer.get("angle", 0))

        is_bold = font_weight == "bold"
        is_italic = font_style == "italic"

        # Parse hex color
        color_hex = color_hex.lstrip("#")
        if len(color_hex) == 6:
            r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
        else:
            r, g, b = 0, 0, 0
        alpha = int(opacity * 255)

        # Check for per-word style spans (proportional mapping from Document AI)
        style_spans = layer.get("styleSpans")

        font = get_font(target_lang, font_size, bold=is_bold, italic=is_italic, font_family=font_family)

        if style_spans:
            # ── Word-by-word styled rendering ──
            word_styles = map_styles_to_words(text, style_spans)
            styled_lines = _wrap_text_styled(word_styles, target_lang, w, font_size)

            if abs(angle) < 1:
                overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                _render_styled_lines(overlay_draw, styled_lines, x, y, w, h,
                                     alignment, (r, g, b), alpha, font_size)
                result = Image.alpha_composite(result, overlay)
            else:
                # Compute actual dimensions for rotated styled text
                line_heights = [max(_line_height(e["font"]) for e in ln) for ln in styled_lines]
                total_h = sum(line_heights)
                space_w = 4
                max_lw = max(
                    sum(e["width"] for e in ln) + space_w * max(0, len(ln) - 1)
                    for ln in styled_lines
                ) if styled_lines else w
                actual_w = max(w, max_lw)
                actual_h = max(h, total_h)
                pad = 8
                canvas_w = actual_w + pad * 2
                canvas_h = actual_h + pad * 2
                txt_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                txt_draw = ImageDraw.Draw(txt_img)
                _render_styled_lines(txt_draw, styled_lines, pad, pad, actual_w, actual_h,
                                     alignment, (r, g, b), alpha, font_size)
                rotated = txt_img.rotate(-angle, resample=Image.BICUBIC, expand=True)
                rw, rh = rotated.size
                paste_x = x + w // 2 - rw // 2
                paste_y = y + h // 2 - rh // 2
                overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
                overlay.paste(rotated, (paste_x, paste_y))
                result = Image.alpha_composite(result, overlay)
        else:
            # ── Uniform style rendering (fallback) ──
            if "\n" in text:
                lines = text.split("\n")
            else:
                lines = _wrap_text(text, font, w)
            lh = _line_height(font)
            total_h = len(lines) * lh
            max_lw = max((font.getbbox(ln)[2] - font.getbbox(ln)[0]) for ln in lines) if lines else w
            actual_w = max(w, max_lw)
            actual_h = max(h, total_h)

            if abs(angle) < 1:
                overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                cy = y + max(0, (h - total_h) // 2)
                for line in lines:
                    line_bbox = font.getbbox(line)
                    line_w = line_bbox[2] - line_bbox[0]
                    if alignment == "left":
                        cx = x
                    elif alignment == "right":
                        cx = x + w - line_w
                    else:
                        cx = x + max(0, (w - line_w) // 2)
                    overlay_draw.text((cx, cy), line, fill=(r, g, b, alpha), font=font)
                    if underline:
                        ul_y = cy + lh - 2
                        ul_thick = max(1, font_size // 14)
                        overlay_draw.line(
                            [(cx, ul_y), (cx + line_w, ul_y)],
                            fill=(r, g, b, alpha), width=ul_thick,
                        )
                    cy += lh
                result = Image.alpha_composite(result, overlay)
            else:
                pad = 8
                canvas_w = actual_w + pad * 2
                canvas_h = actual_h + pad * 2
                txt_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                txt_draw = ImageDraw.Draw(txt_img)
                cy = pad + max(0, (actual_h - total_h) // 2)
                for line in lines:
                    line_bbox = font.getbbox(line)
                    line_w = line_bbox[2] - line_bbox[0]
                    if alignment == "left":
                        cx = pad
                    elif alignment == "right":
                        cx = pad + actual_w - line_w
                    else:
                        cx = pad + max(0, (actual_w - line_w) // 2)
                    txt_draw.text((cx, cy), line, fill=(r, g, b, alpha), font=font)
                    if underline:
                        ul_y = cy + lh - 2
                        ul_thick = max(1, font_size // 14)
                        txt_draw.line(
                            [(cx, ul_y), (cx + line_w, ul_y)],
                            fill=(r, g, b, alpha), width=ul_thick,
                        )
                    cy += lh
                rotated = txt_img.rotate(-angle, resample=Image.BICUBIC, expand=True)
                rw, rh = rotated.size
                paste_x = x + w // 2 - rw // 2
                paste_y = y + h // 2 - rh // 2
                overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
                overlay.paste(rotated, (paste_x, paste_y))
                result = Image.alpha_composite(result, overlay)

    return result.convert("RGB")


# ---------------------------------------------------------------------------
# Helper: PIL Image → base64 data URL
# ---------------------------------------------------------------------------

def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def data_url_to_image(data_url: str) -> Image.Image:
    """Decode a base64 data URL back to a PIL Image."""
    header, b64_data = data_url.split(",", 1)
    image_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/languages")
async def languages():
    langs = _get_languages()
    return JSONResponse(content=langs)


# ── Phase 1a: Detect (OCR + translate, no inpainting) ────────────────────

@app.post("/phase1-detect")
async def phase1_detect(
    file: UploadFile,
    source_lang: str = Form("auto"),
    target_lang: str = Form("en"),
):
    """OCR → translate → return original image + detected region boxes."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    try:
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    regions = detect_text(image_bytes)
    if not regions:
        return JSONResponse(content={
            "original": image_to_data_url(original),
            "width": original.width,
            "height": original.height,
            "regions": [],
        })

    texts = [r["text"] for r in regions]
    translated = translate_texts(texts, source_lang, target_lang)

    region_data = []
    for region, orig_text, trans_text in zip(regions, texts, translated):
        verts = region["vertices"]
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)
        region_data.append({
            "originalText": orig_text,
            "translatedText": trans_text,
            "x": x0, "y": y0,
            "width": x1 - x0, "height": y1 - y0,
            "vertices": verts,
        })

    return JSONResponse(content={
        "original": image_to_data_url(original),
        "width": original.width,
        "height": original.height,
        "regions": region_data,
    })


# ── Phase 1b: Clean (inpaint user-selected regions only) ────────────────

@app.post("/phase1-clean")
async def phase1_clean(request: Request):
    """Inpaint only user-selected regions → return clean image + layer metadata."""
    body = await request.json()
    original_image = body.get("original_image", "")
    selected_regions = body.get("selected_regions", [])
    target_lang = body.get("target_lang", "en")
    mask_mode = body.get("mask_mode", "precise")  # "precise" or "fill"

    try:
        original = data_url_to_image(original_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    sel_regions = selected_regions

    if not sel_regions:
        return JSONResponse(content={
            "original": original_image,
            "clean": original_image,
            "width": original.width,
            "height": original.height,
            "regions": [],
        })

    # Build mask regions — use tight OCR polygon when available, rectangle otherwise
    mask_regions = []
    for r in sel_regions:
        if "vertices" in r and r["vertices"]:
            # Use the original tight OCR polygon vertices
            verts = [tuple(v) for v in r["vertices"]]
        else:
            # Custom drawn box — use rectangle
            x, y, w, h = int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"])
            verts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        mask_regions.append({"vertices": verts})

    # Call Google Document AI for font style detection (if configured)
    # Run in thread pool to avoid blocking the async event loop
    async def _docai_task():
        try:
            img_buf = io.BytesIO()
            original.save(img_buf, format="PNG")
            return await asyncio.to_thread(detect_font_styles_docai, img_buf.getvalue())
        except Exception as e:
            print(f"[DocAI] Skipping: {e}")
            return None

    async def _inpaint_task():
        return await asyncio.to_thread(inpaint_sequential, original, mask_regions, mask_mode)

    # Run Document AI and inpainting concurrently
    docai_styles, inpainted = await asyncio.gather(_docai_task(), _inpaint_task())

    # Build layer metadata
    layer_data = []
    for r in sel_regions:
        x, y = int(r["x"]), int(r["y"])
        w, h = int(r["width"]), int(r["height"])
        trans_text = r.get("translatedText", "")
        orig_text = r.get("originalText", "")

        # Custom drawn box with no text — run OCR on the crop
        if not orig_text and not trans_text and w > 0 and h > 0:
            crop = original.crop((
                max(0, x), max(0, y),
                min(original.width, x + w), min(original.height, y + h)
            ))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crop_regions = detect_text(buf.getvalue())
            if crop_regions:
                orig_text = " ".join(cr["text"] for cr in crop_regions)
                translated = translate_texts([orig_text], "auto", target_lang)
                trans_text = translated[0] if translated else orig_text
            else:
                # No text detected — still include as editable layer with placeholder
                orig_text = "(undetected)"
                trans_text = "(edit text)"
                print(f"[OCR] Custom box {w}x{h} — no text detected, adding as editable layer")

        if not trans_text or w <= 0 or h <= 0:
            continue

        # Detect rotation angle from OCR polygon vertices
        ocr_verts = r.get("vertices")
        angle = 0.0

        if ocr_verts and len(ocr_verts) >= 4:
            angle = _calc_rotation(ocr_verts)

        # Use the axis-aligned bounding box for everything:
        # position, color sampling, font sizing
        # The angle handles the visual rotation in the editor
        sample_verts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        style = analyze_text_style(original, inpainted, sample_verts,
                                   docai_styles=docai_styles, region_text=orig_text)
        color = style["color"]
        is_bold = bool(style["is_bold"])
        is_italic = bool(style["is_italic"])
        is_underline = bool(style["is_underline"])
        font_family = str(style["font_family"])

        # Disable underline for rotated text (false positive from diagonal pixels)
        if abs(angle) > 5:
            is_underline = False

        # Check if detected color is too close to background
        brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
        bg_crop = np.array(inpainted.crop((
            max(0, x), max(0, y),
            min(original.width, x + w), min(original.height, y + h)
        )))
        if bg_crop.size > 0:
            bg_brightness = float(np.mean(bg_crop))
            if abs(brightness - bg_brightness) < 30:
                color = (255, 255, 255) if bg_brightness < 128 else (0, 0, 0)

        alignment = _detect_alignment(x, w, original.width)

        # Use actual text-direction dimensions for rotated text
        abs_angle = abs(angle)
        ocr_verts = r.get("vertices")
        if ocr_verts and len(ocr_verts) >= 4 and abs_angle > 3:
            # Use polygon edge lengths for true text-direction w/h
            rot_w, rot_h = _rotated_box_dims(ocr_verts)
            fit_w = max(rot_w, 10)
            fit_h = max(rot_h, 10)
        elif 60 < abs_angle < 120:
            fit_w, fit_h = h, w
        else:
            fit_w, fit_h = w, h
        # Estimate original font size — prefer Document AI pixel_font_size when available
        word_h = r.get("word_height", 0)
        docai_fs = style.get("_docai_font_size")
        orig_font_size = _estimate_original_font_size(orig_text, fit_h, word_h,
                                                       docai_font_size=docai_fs)
        print(f"[FontSize] '{orig_text[:40]}' → origFS={orig_font_size} (docai={docai_fs}, wordH={word_h}, boxH={fit_h})")
        font_size = _fit_font_size(trans_text, target_lang, fit_w, fit_h,
                                   bold=is_bold, max_size=orig_font_size)
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)

        # Get per-word style spans for proportional mapping
        style_spans = match_region_style_spans(orig_text, docai_styles)
        if style_spans:
            print(f"[StyleSpans] '{orig_text[:40]}' → {len(style_spans)} spans:")
            for sp in style_spans:
                print(f"  [{sp['start']:.2f}-{sp['end']:.2f}] weight={sp.get('fontWeight')} color={sp.get('color')} family={sp.get('fontFamily')}")

        layer_data.append({
            "originalText": orig_text,
            "translatedText": trans_text,
            "x": x, "y": y,
            "width": w, "height": h,
            "fitWidth": fit_w,
            "fitHeight": fit_h,
            "fontSize": font_size,
            "originalFontSize": orig_font_size,
            "color": hex_color,
            "bold": is_bold,
            "italic": is_italic,
            "underline": is_underline,
            "fontFamily": font_family,
            "alignment": alignment,
            "angle": angle,
            "styleSpans": style_spans,
        })

    # ── Post-process: normalize font sizes for spatially grouped regions ──
    _normalize_font_sizes(layer_data, target_lang=target_lang)

    return JSONResponse(content={
        "original": original_image,
        "clean": image_to_data_url(inpainted),
        "width": original.width,
        "height": original.height,
        "regions": layer_data,
    })


# ── Manual Inpaint (touch-up residual artifacts) ─────────────────────────

@app.post("/manual-inpaint")
async def manual_inpaint(request: Request):
    """Inpaint user-drawn rectangular regions on the clean image."""
    body = await request.json()
    image_data = body.get("image", "")
    rectangles = body.get("rectangles", [])

    if not image_data or not rectangles:
        raise HTTPException(400, "image and rectangles required")

    image = data_url_to_image(image_data)

    # Convert rectangles to vertex-based regions for create_mask
    mask_regions = []
    for r in rectangles:
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"])
        verts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        mask_regions.append({"vertices": verts})

    # Use polygon-fill mask (no text-pixel detection) with dilation for full coverage
    mask_arr = np.zeros((image.height, image.width), dtype=np.uint8)
    for mr in mask_regions:
        temp = Image.new("L", image.size, 0)
        ImageDraw.Draw(temp).polygon(mr["vertices"], fill=255)
        temp_arr = np.array(temp)
        # Dilate slightly for clean edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        temp_arr = cv2.dilate(temp_arr, kernel, iterations=1)
        mask_arr = np.maximum(mask_arr, temp_arr)

    mask = Image.fromarray(mask_arr)
    result = inpaint(image, mask)

    return JSONResponse({"clean": image_to_data_url(result)})


# ── Phase 2: Server-side render ──────────────────────────────────────────

@app.post("/phase2-render")
async def phase2_render(request: Request):
    """Render text layers onto clean image server-side for pixel-perfect output."""
    form = await request.form()
    clean_image = form.get("clean_image", "")
    layers_json = form.get("layers_json", "")
    target_lang = form.get("target_lang", "en")

    try:
        base = data_url_to_image(clean_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid clean image data.")

    try:
        layers = json.loads(layers_json)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid layers JSON.")

    result = render_layers_on_image(base, layers, target_lang)
    return JSONResponse(content={"result": image_to_data_url(result)})


# ── Legacy endpoints (kept for backward compat) ──────────────────────────

@app.post("/translate-image")
async def translate_image(
    file: UploadFile,
    source_lang: str = Form("auto"),
    target_lang: str = Form("en"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    try:
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    regions = detect_text(image_bytes)
    if not regions:
        buf = io.BytesIO()
        original.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    texts = [r["text"] for r in regions]
    translated = translate_texts(texts, source_lang, target_lang)
    mask = create_mask(original.size, regions, image=original)
    inpainted = inpaint(original, mask)
    result = render_text(inpainted, original, inpainted, regions, translated, target_lang)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/translate-layers")
async def translate_layers(
    file: UploadFile,
    source_lang: str = Form("auto"),
    target_lang: str = Form("en"),
):
    """Legacy: Return inpainted background + text layer metadata for the editor."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    try:
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    regions = detect_text(image_bytes)
    if not regions:
        return JSONResponse(content={
            "background": image_to_data_url(original),
            "width": original.width,
            "height": original.height,
            "layers": [],
        })

    texts = [r["text"] for r in regions]
    translated = translate_texts(texts, source_lang, target_lang)
    mask = create_mask(original.size, regions, image=original)
    inpainted = inpaint(original, mask)

    layers = []
    for region, orig_text, trans_text in zip(regions, texts, translated):
        verts = region["vertices"]
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)
        box_w = x1 - x0
        box_h = y1 - y0

        if box_w <= 0 or box_h <= 0:
            continue

        color, is_bold = sample_text_color(original, inpainted, verts)
        alignment = _detect_alignment(x0, box_w, original.width)
        font_size = _fit_font_size(trans_text, target_lang, box_w, box_h, bold=is_bold)
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)

        layers.append({
            "originalText": orig_text,
            "translatedText": trans_text,
            "x": x0, "y": y0,
            "width": box_w, "height": box_h,
            "fontSize": font_size,
            "color": hex_color,
            "bold": is_bold,
            "alignment": alignment,
        })

    return JSONResponse(content={
        "background": image_to_data_url(inpainted),
        "width": original.width,
        "height": original.height,
        "layers": layers,
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)
