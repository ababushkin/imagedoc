#!/usr/bin/env python3
"""Convert JPG images in ./input to PDFs in ./output."""

import sys
from pathlib import Path

# Activate local venv if present (so Pillow is found without manual activation)
_venv = Path(__file__).parent / ".venv"
if _venv.exists() and str(_venv / "lib") not in sys.path:
    import site
    pkgs = list((_venv / "lib").glob("python*/site-packages"))
    if pkgs:
        site.addsitedir(str(pkgs[0]))


def convert_images_to_pdfs():
    try:
        from PIL import Image
    except ImportError:
        print("Missing dependency. Install with: pip install Pillow")
        sys.exit(1)

    input_dir = Path("input")
    output_dir = Path("output")

    if not input_dir.exists():
        print(f"Input folder '{input_dir}' not found.")
        sys.exit(1)

    output_dir.mkdir(exist_ok=True)

    images = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg"))
    images += sorted(input_dir.glob("*.JPG")) + sorted(input_dir.glob("*.JPEG"))

    if not images:
        print("No JPG images found in 'input' folder.")
        sys.exit(0)

    converted = 0
    for img_path in images:
        out_path = output_dir / (img_path.stem + ".pdf")
        try:
            with Image.open(img_path) as img:
                rgb = img.convert("RGB")
                rgb.save(out_path, "PDF", resolution=100.0)
            print(f"  {img_path.name} -> {out_path.name}")
            converted += 1
        except Exception as e:
            print(f"  ERROR: {img_path.name}: {e}")

    print(f"\nDone: {converted}/{len(images)} converted.")


if __name__ == "__main__":
    convert_images_to_pdfs()
