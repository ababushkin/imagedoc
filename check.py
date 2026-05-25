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
import re

# ---------------------------------------------------------------------------
# Profile rule tables
# All numeric specs sourced from DFAT (passport) and Home Affairs (visa) docs.
# ---------------------------------------------------------------------------

PROFILES = {
    "passport": {
        # Physical print size range: 35–40 mm wide × 45–50 mm tall (no official DPI/pixel rule)
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "width_px": 413,       # 35 mm × 300 DPI — tool minimum
        "height_px": 531,      # 45 mm × 300 DPI — tool minimum
        "px_tolerance": 10,
        # Valid aspect ratio range: 35/50 (narrowest: 35×50 mm) to 40/45 (widest: 40×45 mm)
        "aspect_min": 35 / 50,   # 0.700
        "aspect_max": 40 / 45,   # 0.889
        # File size band — tool choice; no official rule for a printed photo
        "file_size_min_kb": 50,
        "file_size_max_kb": 2048,
        # Head height: 70–80 % of frame height (face 32–36 mm of 45 mm ≈ 0.711–0.800)
        "head_height_min_frac": 0.70,
        "head_height_max_frac": 0.80,
        "face_height_min_frac": 0.711,
        "face_height_max_frac": 0.800,
        # Background: plain white or light — accepts light grey (L* ≥ 75 ≈ RGB 192,192,192)
        "background_l_min": 75,
        # Colour: must be colour (not greyscale)
        "must_be_colour": True,
        # Print resolution warning threshold: 600 DPI recommended for sharp glossy prints
        "print_warn_width_px": 827,
        "print_warn_height_px": 1063,
        # Auto-fix output: 35 × 45 mm at 600 dpi
        "fix_width_px": 827,
        "fix_height_px": 1063,
    },
    "visa": {
        # Home Affairs digital upload: 354 × 472 px minimum (portrait).
        # The same 35–40 mm × 45–50 mm physical photo is used for both passport and visa,
        # so the valid aspect range is identical. The preferred upload size (1200 × 1600 = 3:4)
        # may differ from the photo aspect; --fix resizes to preferred dimensions.
        "width_px": 354,
        "height_px": 472,
        "px_tolerance": 0,          # exact minimum; larger is accepted
        "aspect_min": 35 / 50,      # 0.700 — narrowest: 35×50 mm
        "aspect_max": 40 / 45,      # 0.889 — widest: 40×45 mm
        "preferred_width_px": 1200,
        "preferred_height_px": 1600,
        "file_size_min_kb": 70,
        "file_size_max_kb": 3584,   # 3.5 MB
        # Head height: 70–80 %
        "head_height_min_frac": 0.70,
        "head_height_max_frac": 0.80,
        "face_height_min_frac": 0.711,
        "face_height_max_frac": 0.800,
        # Background: plain white or light grey (neutral/light, L* ≥ 75)
        "background_l_min": 75,
        "must_be_colour": True,
        # Auto-fix output: preferred upload size
        "fix_width_px": 1200,
        "fix_height_px": 1600,
    },
}

# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

# Human-readable labels and canonical display order for the side-by-side table.
_CHECK_LABELS = {
    "A1": "Format",
    "A2": "File size",
    "A3": "Dimensions / aspect",
    "A4": "Colour mode",
    "B1": "Sharpness",
    "B2": "Brightness",
    "B3": "Background",
    "C1": "Face count",
    "C2": "Head height",
    "C3": "Face centring",
    "C4": "Eyes",
}
_CHECK_ORDER = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C2", "C3", "C4"]


def _worst_status(statuses: list[str]) -> str:
    for s in ("FAIL", "WARN", "PASS"):
        if s in statuses:
            return s
    return "—"


def _findings_by_code(result: "CheckResult") -> dict[str, str]:
    """Return the worst status per check code (e.g. 'A1', 'B3') for a result."""
    code_statuses: dict[str, list[str]] = {}
    for status, msg in result.findings:
        m = re.match(r"([A-Z]\d)", msg)
        if m:
            code = m.group(1)
            code_statuses.setdefault(code, []).append(status)
    return {code: _worst_status(statuses) for code, statuses in code_statuses.items()}


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

    all_passed = all(r.passed for r in results)

    if len(results) > 1:
        names = [r.name.split(": ", 1)[-1].upper() for r in results]
        codes_per = [_findings_by_code(r) for r in results]
        label_w, col_w = 22, 10

        # Comparison table
        print("\n" + " " * label_w + "".join(n.ljust(col_w) for n in names))
        print("-" * (label_w + col_w * len(results)))
        for code in _CHECK_ORDER:
            label = _CHECK_LABELS.get(code, code)
            print(label.ljust(label_w) + "".join(
                codes.get(code, "—").ljust(col_w) for codes in codes_per
            ))
        print("-" * (label_w + col_w * len(results)))
        print("Verdict".ljust(label_w) + "".join(
            ("PASS" if r.passed else "FAIL").ljust(col_w) for r in results
        ))

        # Per-profile details
        for name, result in zip(names, results):
            bar = "─" * max(1, 43 - len(name))
            print(f"\n── {name} details {bar}")
            for s, msg in result.findings:
                print(f"  {s}: {msg}")

        print("\n" + "=" * 60)
        print("  ".join(
            f"{n}: {'PASS' if r.passed else 'FAIL'}"
            for n, r in zip(names, results)
        ))
    else:
        result = results[0]
        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {result.name}")
        for s, msg in result.findings:
            print(f"  {s}: {msg}")
        print("\n" + "=" * 60)
        print(
            "PASS — photo meets all checked requirements."
            if all_passed else
            "FAIL — one or more requirements not met."
        )

    return all_passed


# ---------------------------------------------------------------------------
# Tier A — format, file size, dimensions, colour (Pillow-only)
# ---------------------------------------------------------------------------

def append_tier_a(result: CheckResult, image_path: Path, profile_name: str, profile: dict) -> None:
    """Append Tier A (A1–A4) findings to result; returns early on unreadable file."""
    from PIL import Image

    try:
        with Image.open(image_path) as img:
            pil_format = img.format or ""
            width, height = img.size
            mode = img.mode
    except Exception as exc:
        result.fail(f"A1 cannot open image: {exc}")
        return

    # A1 — format
    suffix = image_path.suffix.lstrip(".").lower()
    ext_ok = suffix in {"jpg", "jpeg", "png"}
    fmt_ok = pil_format.lower() in {"jpeg", "png", "mpo"}
    if ext_ok and fmt_ok:
        if pil_format.upper() == "MPO":
            # MPO is a JPEG-based multi-picture container; most portals accept it
            # as JPEG, but some reject it — convert to plain JPEG if upload fails.
            result.warn(
                f"A1 format MPO (.{suffix}) — JPEG-based container; "
                "accepted but convert to plain JPEG if the portal rejects it"
            )
        elif profile_name == "visa" and pil_format.upper() == "PNG":
            result.warn(
                f"A1 format PNG (.{suffix}) — Home Affairs requires JPEG; "
                "PNG may be rejected by the ImmiAccount portal"
            )
        else:
            result.ok(f"A1 format {pil_format} (.{suffix}) — accepted")
    else:
        result.fail(
            f"A1 format {pil_format or 'unknown'} (.{suffix}) not accepted — "
            "file must be JPEG or PNG"
        )

    # A2 — file size
    size_kb = image_path.stat().st_size / 1024
    min_kb = profile["file_size_min_kb"]
    max_kb = profile["file_size_max_kb"]
    if size_kb < min_kb:
        result.fail(f"A2 file size {size_kb:.0f} KB — below minimum {min_kb} KB")
    elif size_kb > max_kb:
        result.fail(
            f"A2 file size {size_kb:.0f} KB — above maximum {max_kb} KB "
            f"({max_kb / 1024:.1f} MB)"
        )
    else:
        result.ok(f"A2 file size {size_kb:.0f} KB — within {min_kb}–{max_kb} KB")

    # A3 — pixel dimensions and aspect ratio
    min_w = profile["width_px"]
    min_h = profile["height_px"]
    aspect = width / height

    if width < min_w or height < min_h:
        result.fail(f"A3 image {width}×{height} px — below minimum {min_w}×{min_h} px")
    else:
        result.ok(f"A3 image {width}×{height} px — meets minimum {min_w}×{min_h} px")

    aspect_min = profile["aspect_min"]
    aspect_max = profile["aspect_max"]
    if aspect_min <= aspect <= aspect_max:
        result.ok(
            f"A3 aspect ratio {aspect:.3f} — within valid range "
            f"{aspect_min:.3f}–{aspect_max:.3f}"
        )
    else:
        result.fail(
            f"A3 aspect ratio {aspect:.3f} — outside valid range "
            f"{aspect_min:.3f}–{aspect_max:.3f} "
            f"({'too wide' if aspect > aspect_max else 'too tall'})"
        )

    # Warn (not fail) if preferred size is defined and not met
    pref_w = profile.get("preferred_width_px")
    pref_h = profile.get("preferred_height_px")
    if pref_w and pref_h and (width != pref_w or height != pref_h):
        result.warn(
            f"A3 image {width}×{height} px — preferred size is {pref_w}×{pref_h} px "
            "(not required but recommended for best upload quality)"
        )

    # Print resolution warning: only relevant when image meets the minimum
    warn_w = profile.get("print_warn_width_px")
    warn_h = profile.get("print_warn_height_px")
    if (warn_w and warn_h
            and width >= min_w and height >= min_h
            and (width < warn_w or height < warn_h)):
        result.warn(
            f"A3 image {width}×{height} px — below recommended {warn_w}×{warn_h} px "
            "(300 DPI minimum; 600 DPI recommended for sharper print quality)"
        )

    # A4 — colour vs greyscale
    if mode in ("L", "LA"):
        result.fail(f"A4 image is greyscale (mode {mode}) — must be colour")
    else:
        result.ok(f"A4 image is colour (mode {mode})")


# ---------------------------------------------------------------------------
# Tier B — sharpness, brightness, background (OpenCV + numpy)
# ---------------------------------------------------------------------------

# Laplacian variance below this value → image is blurry.
# Calibrated for face ROIs normalised to SHARPNESS_NORM_LONG_SIDE px: a tack-sharp
# AusPost photo (1056×1358, 300 DPI) measures ~44 on the face ROI; threshold must be below that.
SHARPNESS_THRESHOLD = 20.0
# Normalise the face ROI to this long-side limit before measuring Laplacian variance,
# making the threshold independent of source resolution.
SHARPNESS_NORM_LONG_SIDE = 500

# Face-region mean L* (OpenCV 0-255 scale; CIE L* = OpenCV_L / 2.55).
# Below MIN → underexposed; above MAX → overexposed / washed out.
BRIGHTNESS_MIN_L = 64    # ≈ CIE L* 25
BRIGHTNESS_MAX_L = 230   # ≈ CIE L* 90

# Background border-strip thresholds.
BG_BORDER_FRAC = 0.10        # fraction of frame sampled on each edge
BG_SATURATION_MAX = 20.0     # max mean a*b* magnitude (neutral white/grey ≈ 0)
BG_L_VARIANCE_MAX = 400.0    # max L* variance for a uniform background (std ≈ 20)

# Minimum luminance contrast between background and face (visa requirement).
BG_CONTRAST_MIN_L = 30.0     # ≈ CIE ΔL* 12


def append_tier_b(
    result: CheckResult,
    image_path: Path,
    profile_name: str,
    profile: dict,
    face_facts: dict | None,
) -> None:
    """Append Tier B (B1–B3) findings: sharpness, brightness, background."""
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    if img is None:
        result.fail("B1 cannot read image for Tier B checks")
        return

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # OpenCV Lab: L in [0,255], a and b in [0,255] with 128 = neutral.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # Build face ROI once for B1 and B2 (skip if detection found ≠ 1 face).
    face_roi_gray: np.ndarray | None = None
    face_roi_l: np.ndarray | None = None
    face_l_mean: float | None = None
    if face_facts and "face_box_frac" in face_facts:
        xf, yf, wf, hf = face_facts["face_box_frac"]
        x1, y1 = int(xf * w), int(yf * h)
        x2, y2 = int((xf + wf) * w), int((yf + hf) * h)
        if x2 > x1 and y2 > y1:
            face_roi_gray = gray[y1:y2, x1:x2]
            face_roi_l = lab[y1:y2, x1:x2, 0]
            face_l_mean = float(face_roi_l.mean())

    # --- B1: Sharpness ---
    lap_roi = face_roi_gray if face_roi_gray is not None else gray
    # Normalise to SHARPNESS_NORM_LONG_SIDE so the threshold is resolution-independent.
    # Downscaling concentrates edges per pixel; never upscale (that would invent sharpness).
    rh, rw = lap_roi.shape[:2]
    if max(rh, rw) > SHARPNESS_NORM_LONG_SIDE:
        nscale = SHARPNESS_NORM_LONG_SIDE / max(rh, rw)
        lap_roi = cv2.resize(lap_roi, (max(1, round(rw * nscale)), max(1, round(rh * nscale))))
    lap_var = float(cv2.Laplacian(lap_roi, cv2.CV_64F).var())
    if lap_var >= SHARPNESS_THRESHOLD:
        result.ok(f"B1 sharpness OK (Laplacian variance {lap_var:.0f})")
    else:
        result.fail(
            f"B1 image too blurry (Laplacian variance {lap_var:.0f} "
            f"< threshold {SHARPNESS_THRESHOLD:.0f}) "
            "— re-take with a steady camera; do not sharpen in post-processing"
        )

    # --- B2: Brightness/exposure ---
    l_roi = face_roi_l if face_roi_l is not None else lab[:, :, 0]
    mean_l = float(l_roi.mean())
    if mean_l < BRIGHTNESS_MIN_L:
        result.fail(
            f"B2 underexposed (mean L {mean_l:.0f}/255 "
            f"< minimum {BRIGHTNESS_MIN_L}) "
            "— re-take with more light or use flash; editing exposure is not permitted"
        )
    elif mean_l > BRIGHTNESS_MAX_L:
        result.fail(
            f"B2 overexposed (mean L {mean_l:.0f}/255 "
            f"> maximum {BRIGHTNESS_MAX_L}) "
            "— re-take away from direct light; editing exposure is not permitted"
        )
    else:
        result.ok(f"B2 brightness OK (mean L {mean_l:.0f}/255)")

    # --- B3: Background uniformity ---
    bh = max(1, int(h * BG_BORDER_FRAC))
    bw = max(1, int(w * BG_BORDER_FRAC))
    # When the face box is known, sample only the top strip — the one region guaranteed
    # to be background in a correctly framed portrait.  The left/right/bottom strips are
    # contaminated by hair, ears, and clothing in a photo that fills the frame as required.
    # Without face information fall back to the full 4-edge border.
    if face_facts and "face_box_frac" in face_facts:
        border_pixels = lab[:bh, :, :].reshape(-1, 3).astype(float)
    else:
        border_pixels = np.concatenate([
            lab[:bh, :, :].reshape(-1, 3),
            lab[h - bh:, :, :].reshape(-1, 3),
            lab[:, :bw, :].reshape(-1, 3),
            lab[:, w - bw:, :].reshape(-1, 3),
        ], axis=0).astype(float)

    l_vals = border_pixels[:, 0]
    ab_shifted = border_pixels[:, 1:3] - 128.0   # centre around neutral
    bg_l_mean = float(l_vals.mean())
    bg_l_var = float(l_vals.var())
    bg_saturation = float(np.sqrt((ab_shifted ** 2).sum(axis=1)).mean())

    bg_l_min_cie = profile.get("background_l_min", 90)
    bg_l_min_ocv = bg_l_min_cie * 2.55

    b3_ok = True
    if bg_l_mean < bg_l_min_ocv:
        result.fail(
            f"B3 background too dark (border mean L {bg_l_mean:.0f}/255, "
            f"need ≥ {bg_l_min_ocv:.0f} for CIE L* ≥ {bg_l_min_cie}) "
            "— re-take against a plain white or light-grey background"
        )
        b3_ok = False
    if bg_saturation > BG_SATURATION_MAX:
        result.fail(
            f"B3 background has colour/tint "
            f"(border a*b* {bg_saturation:.1f} > {BG_SATURATION_MAX:.0f}) "
            "— re-take against a plain neutral-white background"
        )
        b3_ok = False
    if bg_l_var > BG_L_VARIANCE_MAX:
        result.fail(
            f"B3 background not uniform "
            f"(border L* variance {bg_l_var:.0f} > {BG_L_VARIANCE_MAX:.0f}) "
            "— re-take against a plain background with no shadows or patterns"
        )
        b3_ok = False

    if b3_ok:
        result.ok(
            f"B3 background OK "
            f"(L {bg_l_mean:.0f}/255, saturation {bg_saturation:.1f}, "
            f"variance {bg_l_var:.0f})"
        )

    # Both specs require the face to be distinct from the background.
    if face_l_mean is not None:
        contrast = abs(bg_l_mean - face_l_mean)
        if contrast >= BG_CONTRAST_MIN_L:
            result.ok(f"B3 face/background contrast OK (ΔL {contrast:.0f})")
        else:
            result.fail(
                f"B3 face/background contrast insufficient "
                f"(ΔL {contrast:.0f} < {BG_CONTRAST_MIN_L:.0f}) "
                "— face blends into background"
            )


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

# The Haar frontal-face box spans roughly the forehead to the chin, which is
# shorter than the full crown-to-chin head height the specs measure. This factor
# converts detected face-box height to an estimated head height. Calibrated by
# cropping a reference portrait to a known 75%-of-frame head and measuring the
# resulting Haar box at ~58% of frame → 0.75 / 0.58 ≈ 1.30. Derived from a single
# reference face; crown position varies with hairstyle, so re-validate against
# more portraits if outputs frame the head too high or low.
HEAD_FACE_RATIO = 1.30


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
    facts["face_box_frac"] = (fx / ww, fy / wh, fw / ww, fh / wh)

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

    hf = facts["height_frac"]  # Haar face-box (brow-to-chin) as fraction of frame
    estimated_head_frac = hf * HEAD_FACE_RATIO  # crown-to-chin estimate
    lo, hi = profile["head_height_min_frac"], profile["head_height_max_frac"]
    if lo <= estimated_head_frac <= hi:
        result.ok(f"C2 estimated head height {estimated_head_frac:.0%} of frame (target {lo:.0%}–{hi:.0%})")
    else:
        result.fail(
            f"C2 estimated head height {estimated_head_frac:.0%} of frame, "
            f"outside target {lo:.0%}–{hi:.0%} — crop/scale needed"
        )

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


def _fix_output_path(image_path: Path, profile_name: str, out: Path | None, multi_profile: bool) -> Path:
    """Derive the output path for a fixed image."""
    if out is not None and not multi_profile:
        return out
    suffix = image_path.suffix or ".jpg"
    tag = f"_{profile_name}_fixed" if multi_profile else "_fixed"
    return image_path.parent / f"{image_path.stem}{tag}{suffix}"


def _save_jpeg_within_size(img_bgr, out_path: Path, min_kb: float, max_kb: float) -> float:
    """Save JPEG at the highest quality with file size in [min_kb, max_kb].

    Falls back to quality 95 if no quality in [1, 95] lands in the band.
    Returns actual file size in KB.
    """
    import cv2

    def _enc(q: int) -> tuple[bytes, float]:
        ok, raw = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        return (raw.tobytes(), len(raw) / 1024) if ok else (b"", 0.0)

    data, size_kb = _enc(95)

    if min_kb <= size_kb <= max_kb:
        out_path.write_bytes(data)
        return size_kb

    if size_kb > max_kb:
        # Binary search for the highest quality whose file size is ≤ max_kb.
        lo, hi, best_data, best_size = 1, 94, b"", 0.0
        while lo <= hi:
            q = (lo + hi) // 2
            d, s = _enc(q)
            if s <= max_kb:
                best_data, best_size = d, s
                lo = q + 1
            else:
                hi = q - 1
        if best_data and best_size >= min_kb:
            out_path.write_bytes(best_data)
            return best_size

    # Fallback: quality 95 — A2 will report the actual out-of-band size.
    out_path.write_bytes(data)
    return size_kb


def crop_to_profile(image_path: Path, profile: dict, face_facts: dict, out_path: Path) -> bool:
    """Crop and scale source image to profile fix dimensions with head in the target band.

    Returns True on success. The crop centres the face horizontally and places the
    estimated crown with balanced top/bottom margins. Regions outside the source
    image are padded with white.
    """
    import cv2
    import numpy as np

    if "face_box_frac" not in face_facts:
        return False

    img = cv2.imread(str(image_path))
    if img is None:
        return False

    src_h, src_w = img.shape[:2]
    xf, yf, wf, hf = face_facts["face_box_frac"]

    # Face box in source pixels
    fx = xf * src_w
    fy = yf * src_h
    fw = wf * src_w
    fh = hf * src_h

    # Estimated head bounds using the calibration factor
    crown_y = fy - (HEAD_FACE_RATIO - 1.0) * fh   # crown above the face-box top
    chin_y = fy + fh                                 # chin at the face-box bottom
    head_h_src = chin_y - crown_y                    # = fh * HEAD_FACE_RATIO
    face_cx = fx + fw / 2.0

    # Target: head at the centre of the 70–80% band
    head_frac = (profile["head_height_min_frac"] + profile["head_height_max_frac"]) / 2.0
    out_w = profile["fix_width_px"]
    out_h = profile["fix_height_px"]

    # Crop region in source pixels: scale so head_h_src maps to head_frac * out_h
    # crop_h / out_h = head_h_src / (head_frac * out_h)  →  crop_h = head_h_src / head_frac
    crop_h = head_h_src / head_frac
    crop_w = crop_h * out_w / out_h  # maintain output aspect ratio

    # Balanced vertical margin: crown at (1 - head_frac) / 2 from top of crop
    top_margin = (1.0 - head_frac) / 2.0 * crop_h
    crop_y1 = crown_y - top_margin
    crop_x1 = face_cx - crop_w / 2.0

    # Convert to integer pixel coordinates
    cx1 = int(round(crop_x1))
    cy1 = int(round(crop_y1))
    cw = int(round(crop_w))
    ch = int(round(crop_h))

    # White canvas; copy the source region into the correct position
    canvas = np.full((ch, cw, 3), 255, dtype=np.uint8)
    sx1 = max(0, cx1)
    sy1 = max(0, cy1)
    sx2 = min(src_w, cx1 + cw)
    sy2 = min(src_h, cy1 + ch)
    if sx2 > sx1 and sy2 > sy1:
        dx1 = sx1 - cx1
        dy1 = sy1 - cy1
        canvas[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = img[sy1:sy2, sx1:sx2]

    out_img = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        _save_jpeg_within_size(
            out_img, out_path,
            profile.get("file_size_min_kb", 0),
            profile.get("file_size_max_kb", float("inf")),
        )
    else:
        cv2.imwrite(str(out_path), out_img)
    return True


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

        append_tier_a(r, image_path, profile_name, profile)
        append_tier_b(r, image_path, profile_name, profile, face_facts)
        append_face_geometry(r, face_facts, profile)

        results.append(r)

    overall = print_report(results, str(image_path))

    if fix and face_facts:
        multi = len(profiles) > 1
        fixed: list[tuple[str, Path]] = []
        for profile_name in profiles:
            profile = PROFILES[profile_name]
            fixed_path = _fix_output_path(image_path, profile_name, out, multi)
            if crop_to_profile(image_path, profile, face_facts, fixed_path):
                print(f"\nFixed ({profile_name}): {fixed_path}")
                fixed.append((profile_name, fixed_path))
            else:
                print(f"Auto-fix failed for {profile_name} — no face detected")

        for profile_name, fixed_path in fixed:
            run_checks(fixed_path, [profile_name], fix=False, out=None)

    return overall


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
