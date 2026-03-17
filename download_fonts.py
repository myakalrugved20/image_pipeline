"""Download Google Noto Sans fonts needed for text rendering."""
import os
import urllib.request
from pathlib import Path

FONT_DIR = Path(__file__).resolve().parent / "fonts"

FONTS = {
    "NotoSans-Regular.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf"
    ),
    "NotoSans-Italic.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans-Italic%5Bwdth%2Cwght%5D.ttf"
    ),
    "NotoSerif-Regular.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/notoserif/NotoSerif%5Bwdth%2Cwght%5D.ttf"
    ),
    "NotoSansMono-Regular.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/notosansmono/NotoSansMono%5Bwdth%2Cwght%5D.ttf"
    ),
    "NotoSansArabic-Regular.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/notosansarabic/NotoSansArabic%5Bwdth%2Cwght%5D.ttf"
    ),
}

# Bold variants (same variable font files support bold via weight axis,
# but Pillow can't select weight from variable fonts, so we download
# static bold builds where available)
BOLD_FONTS = {
    "NotoSans-Bold.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans-Bold.ttf"
    ),
}

# CJK font from a different repo (large file ~16 MB)
CJK_URL = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
CJK_NAME = "NotoSansCJKsc-Regular.otf"


def download(url: str, dest: Path):
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    print(f"  Downloading {dest.name} …")
    try:
        urllib.request.urlretrieve(url, str(dest))
        print(f"  Saved {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
    except urllib.error.HTTPError as e:
        print(f"  Warning: could not download {dest.name} ({e}) — skipping")


def main():
    FONT_DIR.mkdir(exist_ok=True)
    print(f"Font directory: {FONT_DIR}\n")

    for name, url in FONTS.items():
        download(url, FONT_DIR / name)

    for name, url in BOLD_FONTS.items():
        download(url, FONT_DIR / name)

    download(CJK_URL, FONT_DIR / CJK_NAME)

    print("\nDone! All fonts are ready.")


if __name__ == "__main__":
    main()
