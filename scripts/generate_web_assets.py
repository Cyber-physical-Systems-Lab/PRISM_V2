#!/usr/bin/env python3
"""Generate web-ready images from assets/figures PNGs.

Creates `assets/figures/web/` and writes WebP full-size and 400px-wide
thumbnail variants while preserving original files.

Usage:
  python scripts/generate_web_assets.py

Requires: Pillow
  pip install Pillow
"""
from pathlib import Path
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"
OUT_DIR = FIG_DIR / "web"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def process_png(p: Path):
    name = p.stem
    im = Image.open(p)
    # convert to RGB if needed
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")

    full_out = OUT_DIR / f"{name}.webp"
    thumb_out = OUT_DIR / f"{name}_thumb.webp"

    # full-size: encode as WebP quality 85
    im.save(full_out, format="WEBP", quality=85, method=6)

    # thumbnail: width 400px
    w = 400
    ratio = w / im.width
    h = max(1, int(im.height * ratio))
    thumb = im.resize((w, h), Image.LANCZOS)
    thumb.save(thumb_out, format="WEBP", quality=80, method=6)


def main():
    pngs = sorted(FIG_DIR.glob("*.png"))
    if not pngs:
        print("No PNG files found in", FIG_DIR)
        return
    for p in pngs:
        print("Processing", p.name)
        try:
            process_png(p)
        except Exception as e:
            print("Failed to process", p, e)


if __name__ == "__main__":
    main()
