import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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

# Load LaMa once (heavy model) — force CPU when CUDA is unavailable
print("Loading LaMa inpainting model …")
import torch
if not torch.cuda.is_available():
    _orig_jit_load = torch.jit.load
    torch.jit.load = lambda f, **kw: _orig_jit_load(f, **{**kw, "map_location": "cpu"})
    lama = SimpleLama()
    torch.jit.load = _orig_jit_load
else:
    lama = SimpleLama()
print("LaMa ready.")

# Google Vision client
vision_client = vision.ImageAnnotatorClient()

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
                for word in paragraph.words:
                    word_text = "".join(s.text for s in word.symbols)
                    text += word_text + " "
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

                print(f"[OCR KEEP] {box_w}x{box_h} area={area} — '{text[:60]}'")
                regions.append({"text": text, "vertices": vertices})

    return regions


# ── Translation ───────────────────────────────────────────────────────────

def translate_texts(texts: list[str], source: str, target: str) -> list[str]:
    """Translate a list of strings via Google Translate."""
    if not texts:
        return []
    try:
        translator = GoogleTranslator(source=source, target=target)
        results = translator.translate_batch(texts)
        return [r if r else t for r, t in zip(results, texts)]
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
    result = lama(image.convert("RGB"), mask.convert("L"))
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
    # Italic text has a consistent rightward slant with good correlation
    return abs(slope) > 0.12 and abs(r_value) > 0.5


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
    for row in bottom_region:
        run = np.sum(row >= threshold)
        if run > box_w * 0.6:
            return True

    return False


def analyze_text_style(
    original: Image.Image,
    inpainted: Image.Image,
    vertices: list[tuple[int, int]],
) -> dict:
    """Comprehensive text style analysis: color, bold, italic, underline, font family."""
    color, is_bold = sample_text_color(original, inpainted, vertices)
    is_italic = detect_italic(original, inpainted, vertices)
    is_underline = detect_underline(original, inpainted, vertices)
    font_family = detect_font_family(original, vertices)

    return {
        "color": color,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "is_underline": is_underline,
        "font_family": font_family,
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

def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-based wrapping using Pillow's getbbox for pixel accuracy."""
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


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    """Compute line height from font metrics."""
    bbox = font.getbbox("Ayg|")
    return (bbox[3] - bbox[1]) + 2


def _fit_font_size(text: str, target_lang: str, box_w: int, box_h: int, bold: bool = False) -> int:
    """Binary search for the largest font size where text fits the box."""
    lo, hi = 6, max(box_h, 10)
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

        font = get_font(target_lang, font_size, bold=is_bold, italic=is_italic, font_family=font_family)
        # If text contains newlines, it was pre-wrapped by the editor — use as-is
        if "\n" in text:
            lines = text.split("\n")
        else:
            lines = _wrap_text(text, font, w)
        lh = _line_height(font)
        total_h = len(lines) * lh
        # Actual max line width (may exceed bounding box w with server fonts)
        max_lw = max((font.getbbox(ln)[2] - font.getbbox(ln)[0]) for ln in lines) if lines else w
        actual_w = max(w, max_lw)
        actual_h = max(h, total_h)

        if abs(angle) < 1:
            # No rotation — draw directly on full overlay (fast path)
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
            # Rotated text — render upright on a sized canvas, rotate, paste
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
            # Rotate around center (negative because Pillow rotates CCW)
            rotated = txt_img.rotate(-angle, resample=Image.BICUBIC, expand=True)
            # Paste centered at the bounding box center
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
    return templates.TemplateResponse("index.html", {"request": request})


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

    # Inpaint regions — pass mask_mode to control masking strategy
    inpainted = inpaint_sequential(original, mask_regions, mask_mode=mask_mode)

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
        style = analyze_text_style(original, inpainted, sample_verts)
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
        font_size = _fit_font_size(trans_text, target_lang, fit_w, fit_h, bold=is_bold)
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)

        layer_data.append({
            "originalText": orig_text,
            "translatedText": trans_text,
            "x": x, "y": y,
            "width": w, "height": h,
            "fitWidth": fit_w,
            "fitHeight": fit_h,
            "fontSize": font_size,
            "color": hex_color,
            "bold": is_bold,
            "italic": is_italic,
            "underline": is_underline,
            "fontFamily": font_family,
            "alignment": alignment,
            "angle": angle,
        })

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
