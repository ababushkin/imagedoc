#!/usr/bin/env python3
"""Check a photo against AU passport and/or visa photo specs."""

import sys
from pathlib import Path

# Activate local venv if present (so cv2/numpy/Pillow are found without manual activation)
_venv = Path(__file__).parent / ".venv"
if _venv.exists() and str(_venv / "lib") not in sys.path:
    import site
    pkgs = list((_venv / "lib").glob("python*/site-packages"))
    if pkgs:
        site.addsitedir(str(pkgs[0]))

import argparse

# ---------------------------------------------------------------------------
# Profile rule tables
# All numeric specs sourced from DFAT (passport) and Home Affairs (visa) docs.
# ---------------------------------------------------------------------------

PROFILES = {
    "passport": {
        # Physical print size: 35 × 45 mm at 300 dpi → 413 × 531 px (±5 px tolerance)
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "width_px": 413,
        "height_px": 531,
        "px_tolerance": 10,
        # Aspect ratio: width / height = 35/45 ≈ 0.778
        "aspect_ratio": 35 / 45,
        "aspect_tolerance": 0.02,
        # File size band (JPEG bytes)
        "file_size_min_kb": 50,
        "file_size_max_kb": 2048,
        # Head height: 70–80 % of frame height
        "head_height_min_frac": 0.70,
        "head_height_max_frac": 0.80,
        # Face height (chin to crown): 32–36 mm out of 45 mm ≈ 0.711–0.800
        "face_height_min_frac": 0.711,
        "face_height_max_frac": 0.800,
        # Background: plain white/off-white (L* > 90 in CIE Lab)
        "background_l_min": 90,
        # Colour: must be colour (not greyscale)
        "must_be_colour": True,
    },
    "visa": {
        # Home Affairs digital upload: 350 × 350 px minimum, square crop
        "width_px": 350,
        "height_px": 350,
        "px_tolerance": 0,          # exact minimum; larger is accepted
        "aspect_ratio": 1.0,
        "aspect_tolerance": 0.02,
        "file_size_min_kb": 10,
        "file_size_max_kb": 1000,
        # Head height: 70–80 %
        "head_height_min_frac": 0.70,
        "head_height_max_frac": 0.80,
        "face_height_min_frac": 0.711,
        "face_height_max_frac": 0.800,
        "background_l_min": 90,
        "must_be_colour": True,
    },
}

# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, name: str):
        self.name = name
        self.findings: list[tuple[str, str]] = []  # (PASS|FAIL|WARN, message)

    def ok(self, msg: str):
        self.findings.append(("PASS", msg))

    def fail(self, msg: str):
        self.findings.append(("FAIL", msg))

    def warn(self, msg: str):
        self.findings.append(("WARN", msg))

    @property
    def passed(self) -> bool:
        return all(status != "FAIL" for status, _ in self.findings)


def print_report(results: list[CheckResult], image_path: str) -> bool:
    """Print PASS/FAIL report; return True if all checks passed."""
    print(f"\nPhoto compliance check: {image_path}")
    print("=" * 60)
    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {result.name}")
        for s, msg in result.findings:
            indent = "  "
            print(f"{indent}{s}: {msg}")
        if not result.passed:
            all_passed = False
    print("\n" + "=" * 60)
    overall = "PASS — photo meets all checked requirements." if all_passed else "FAIL — one or more requirements not met."
    print(overall)
    return all_passed


# ---------------------------------------------------------------------------
# Tier C — face geometry (OpenCV Haar cascades, bundled, fully offline)
# ---------------------------------------------------------------------------

# Detection is run on a downscaled copy so runtime and minNeighbors behaviour
# stay consistent regardless of the source resolution. All reported quantities
# are fractions of the frame, so they are unaffected by the downscale.
DETECT_MAX_DIM = 1000

# Ignore face boxes smaller than this fraction of the shorter frame side — keeps
# the cascade from latching onto background clutter as a tiny "face".
FACE_MIN_FRAC = 0.10

# The face centre may sit within ±this fraction of the frame width from the middle.
CENTRE_TOLERANCE = 0.10


def detect_face_and_eyes(image_path: Path) -> dict | None:
    """Run the frontal-face and eye cascades once on an image.

    Returns the geometric facts the per-profile checks need, or None if the
    image could not be read. The face box height and centre are measured against
    the detected-face bounding box (brow-to-chin), which understates the full
    crown-to-chin head height; the head band is the target after cropping, so a
    raw, uncropped portrait is expected to fall short of it here.
    """
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = DETECT_MAX_DIM / max(h, w)
    work = cv2.resize(img, (round(w * scale), round(h * scale))) if scale < 1 else img
    gray_plain = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    # Histogram equalisation makes frontal-face detection robust to lighting, but
    # it suppresses the eye cascade — so faces use the equalised image, eyes the plain one.
    gray = cv2.equalizeHist(gray_plain)
    wh, ww = gray.shape[:2]

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    min_side = int(min(ww, wh) * FACE_MIN_FRAC)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
    )
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    facts: dict = {"n_faces": len(faces)}
    if not faces:
        return facts

    fx, fy, fw, fh = faces[0]
    facts["height_frac"] = fh / wh
    facts["centre_x_frac"] = (fx + fw / 2) / ww

    # Eyes are expected in the upper half of an upright frontal face; restricting
    # the search there rejects nostril/mouth false positives and confirms orientation.
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    roi = gray_plain[fy:fy + fh // 2, fx:fx + fw]
    eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5)
    facts["n_eyes_upper"] = len(eyes)
    return facts


def append_face_geometry(result: CheckResult, facts: dict | None, profile: dict) -> None:
    """Append the four Tier C face-geometry findings to a profile's result."""
    if facts is None:
        result.fail("C1 could not read image for face detection")
        return

    n = facts["n_faces"]
    if n == 0:
        result.fail("C1 no frontal face detected — not a portrait (document scan, or face too small/turned)")
        return
    if n > 1:
        result.fail(f"C1 {n} faces detected — a passport photo must contain exactly one")
        return
    result.ok("C1 exactly one frontal face detected")

    hf = facts["height_frac"]
    lo, hi = profile["head_height_min_frac"], profile["head_height_max_frac"]
    if lo <= hf <= hi:
        result.ok(f"C2 face height {hf:.0%} of frame (target {lo:.0%}–{hi:.0%})")
    else:
        result.fail(f"C2 face height {hf:.0%} of frame, outside target {lo:.0%}–{hi:.0%} — crop/scale needed")

    cx = facts["centre_x_frac"]
    offset = abs(cx - 0.5)
    if offset <= CENTRE_TOLERANCE:
        result.ok(f"C3 face horizontally centred (centre at {cx:.0%} of width)")
    else:
        result.fail(f"C3 face off-centre (centre at {cx:.0%} of width, must be 50% ±{CENTRE_TOLERANCE:.0%})")

    eyes = facts["n_eyes_upper"]
    if eyes >= 2:
        result.ok(f"C4 {eyes} eyes detected in upper half of face")
    elif eyes == 1:
        result.warn("C4 only one eye detected in upper half of face — head may be tilted or partly obscured")
    else:
        result.fail("C4 no eyes detected in upper half of face — face may be obscured, tilted, or a false detection")


# ---------------------------------------------------------------------------
# Stub checks (Tier A/B implemented in subsequent issues)
# ---------------------------------------------------------------------------

def run_checks(image_path: Path, profiles: list[str], fix: bool, out: Path | None) -> bool:
    """Run all checks for the selected profiles; return True if all pass."""
    try:
        import cv2  # noqa: F401  (verified at import time)
        import numpy  # noqa: F401
        from PIL import Image
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    results: list[CheckResult] = []

    # Face detection is profile-independent — run the cascades once and reuse.
    face_facts = detect_face_and_eyes(image_path)

    for profile_name in profiles:
        profile = PROFILES[profile_name]
        r = CheckResult(f"Profile: {profile_name}")

        # Tier A/B checks are stubs — will be filled in by their own issues.
        r.ok("[stub] format, size, dimensions — not yet implemented")
        r.ok("[stub] sharpness, brightness, background — not yet implemented")
        append_face_geometry(r, face_facts, profile)

        results.append(r)

    return print_report(results, str(image_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check a photo against AU passport/visa photo specs.",
    )
    parser.add_argument("image", help="Path to the image file to check.")
    parser.add_argument(
        "--profile",
        choices=["passport", "visa", "both"],
        default="both",
        help="Which spec profile(s) to check against (default: both).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to auto-fix issues and write corrected image.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path for fixed image (implies --fix).",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    profiles = ["passport", "visa"] if args.profile == "both" else [args.profile]
    fix = args.fix or args.out is not None

    passed = run_checks(image_path, profiles, fix=fix, out=args.out)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
