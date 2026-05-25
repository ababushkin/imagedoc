"""Tests for check.py — covers Tier A, B, and C rules plus profile sanity checks."""

import copy
import pytest
import numpy as np
import cv2
from pathlib import Path
from PIL import Image as PILImage

from check import (
    PROFILES,
    CheckResult,
    HEAD_FACE_RATIO,
    SHARPNESS_THRESHOLD,
    SHARPNESS_NORM_LONG_SIDE,
    append_tier_a,
    append_tier_b,
    append_face_geometry,
    print_report,
    crop_to_profile,
    _save_jpeg_within_size,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fails(result: CheckResult) -> list[str]:
    return [msg for s, msg in result.findings if s == "FAIL"]

def warns(result: CheckResult) -> list[str]:
    return [msg for s, msg in result.findings if s == "WARN"]

def passes(result: CheckResult) -> list[str]:
    return [msg for s, msg in result.findings if s == "PASS"]

def has_fail_tagged(result: CheckResult, tag: str) -> bool:
    return any(tag in msg for msg in fails(result))

def has_warn_tagged(result: CheckResult, tag: str) -> bool:
    return any(tag in msg for msg in warns(result))

# ---------------------------------------------------------------------------
# Image factories
# ---------------------------------------------------------------------------

def make_jpeg(tmp_path: Path, width: int, height: int,
              color=(255, 255, 255), quality: int = 85,
              name: str | None = None) -> Path:
    path = tmp_path / (name or f"img_{width}x{height}.jpg")
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path

def make_png(tmp_path: Path, width: int, height: int,
             color=(255, 255, 255), name: str | None = None) -> Path:
    path = tmp_path / (name or f"img_{width}x{height}.png")
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), arr)
    return path

def make_grayscale_jpeg(tmp_path: Path, width: int, height: int,
                        value: int = 128, name: str | None = None) -> Path:
    path = tmp_path / (name or f"gray_{width}x{height}.jpg")
    img = PILImage.new("L", (width, height), value)
    img.save(str(path), "JPEG")
    return path

def make_noisy_jpeg(tmp_path: Path, width: int, height: int,
                    name: str | None = None) -> Path:
    """Create a JPEG with random noise — high Laplacian variance (sharp)."""
    path = tmp_path / (name or f"noisy_{width}x{height}.jpg")
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), arr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path

def make_sine_jpeg(tmp_path: Path, width: int, height: int,
                   amplitude: float = 20.0, wavelength: float = 10.0,
                   name: str | None = None) -> Path:
    """Sinusoidal horizontal stripe pattern.

    Discrete Laplacian variance ≈ 2*A² * (cos(2π/λ)-1)² ≈ 29 for A=20, λ=10.
    This is above SHARPNESS_THRESHOLD (20) but below the old value of 100.
    """
    x = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    vals = 128.0 + amplitude * np.sin(2 * np.pi * x / wavelength)
    arr = np.clip(vals, 0, 255).astype(np.uint8)
    arr = np.stack([arr, arr, arr], axis=2)
    path = tmp_path / (name or f"sine_{width}x{height}.jpg")
    cv2.imwrite(str(path), arr, [cv2.IMWRITE_JPEG_QUALITY, 99])
    return path


def make_portrait_tight_jpeg(tmp_path: Path, width: int, height: int,
                              name: str | None = None) -> Path:
    """Portrait with white top strip (background) and grey body/hair everywhere else.

    The top BG_BORDER_FRAC strip is white; the rest is mid-grey so the 4-edge
    border strips contain a mix → high variance → old B3 would fail.
    With face-aware sampling (top strip only) the background is uniformly white → passes.
    """
    arr = np.full((height, width, 3), 100, dtype=np.uint8)   # mid-grey subject
    bh = max(1, int(height * 0.10))
    arr[:bh, :] = 250                                          # white background at top
    path = tmp_path / (name or f"portrait_tight_{width}x{height}.jpg")
    cv2.imwrite(str(path), arr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def make_contrast_jpeg(tmp_path: Path, width: int, height: int,
                       border_gray: int, face_gray: int,
                       name: str | None = None) -> Path:
    """Create JPEG with uniform border (border_gray) and a center rectangle (face_gray)."""
    path = tmp_path / (name or f"contrast_b{border_gray}_f{face_gray}.jpg")
    arr = np.full((height, width, 3), border_gray, dtype=np.uint8)
    fy1, fy2 = int(height * 0.3), int(height * 0.7)
    fx1, fx2 = int(width * 0.3), int(width * 0.7)
    arr[fy1:fy2, fx1:fx2] = face_gray
    cv2.imwrite(str(path), arr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


# ---------------------------------------------------------------------------
# Profile sanity checks
# ---------------------------------------------------------------------------

class TestProfileConstants:
    def test_passport_aspect_range_covers_35x45(self):
        asp = 35 / 45
        assert PROFILES["passport"]["aspect_min"] <= asp <= PROFILES["passport"]["aspect_max"]

    def test_passport_aspect_range_covers_40x50(self):
        asp = 40 / 50
        assert PROFILES["passport"]["aspect_min"] <= asp <= PROFILES["passport"]["aspect_max"]

    def test_passport_aspect_range_boundaries(self):
        p = PROFILES["passport"]
        assert abs(p["aspect_min"] - 35 / 50) < 1e-6
        assert abs(p["aspect_max"] - 40 / 45) < 1e-6

    def test_passport_background_threshold_accepts_light_grey(self):
        # L* 75 threshold: RGB (192,192,192) ≈ CIE L* 78.7 — must be above threshold
        assert PROFILES["passport"]["background_l_min"] <= 75

    def test_visa_background_threshold_accepts_light_grey(self):
        assert PROFILES["visa"]["background_l_min"] <= 75

    def test_passport_has_print_warn_thresholds(self):
        p = PROFILES["passport"]
        assert "print_warn_width_px" in p
        assert "print_warn_height_px" in p
        assert p["print_warn_width_px"] > p["width_px"]
        assert p["print_warn_height_px"] > p["height_px"]

    def test_passport_no_legacy_aspect_keys(self):
        p = PROFILES["passport"]
        assert "aspect_ratio" not in p
        assert "aspect_tolerance" not in p

    def test_visa_head_height_band_matches_passport(self):
        assert PROFILES["visa"]["head_height_min_frac"] == PROFILES["passport"]["head_height_min_frac"]
        assert PROFILES["visa"]["head_height_max_frac"] == PROFILES["passport"]["head_height_max_frac"]

    def test_visa_aspect_range_covers_35x45(self):
        asp = 35 / 45
        assert PROFILES["visa"]["aspect_min"] <= asp <= PROFILES["visa"]["aspect_max"]

    def test_visa_aspect_range_matches_passport(self):
        assert PROFILES["visa"]["aspect_min"] == PROFILES["passport"]["aspect_min"]
        assert PROFILES["visa"]["aspect_max"] == PROFILES["passport"]["aspect_max"]

    def test_visa_no_legacy_aspect_keys(self):
        v = PROFILES["visa"]
        assert "aspect_ratio" not in v, "aspect_ratio key must be removed from visa profile"
        assert "aspect_tolerance" not in v, "aspect_tolerance key must be removed from visa profile"

    def test_sharpness_threshold_calibrated_for_real_portraits(self):
        # AusPost photo (tack-sharp, 300 DPI) measured ~44; threshold must be below that.
        assert SHARPNESS_THRESHOLD <= 44.0
        assert SHARPNESS_NORM_LONG_SIDE >= 400


# ---------------------------------------------------------------------------
# Tier A
# ---------------------------------------------------------------------------

class TestTierA_Format:
    def test_jpeg_passes_passport(self, tmp_path):
        path = make_jpeg(tmp_path, 413, 531)
        r = CheckResult("test")
        append_tier_a(r, path, "passport", PROFILES["passport"])
        assert not has_fail_tagged(r, "A1")

    def test_jpeg_passes_visa(self, tmp_path):
        path = make_jpeg(tmp_path, 354, 472)
        r = CheckResult("test")
        append_tier_a(r, path, "visa", PROFILES["visa"])
        assert not has_fail_tagged(r, "A1")
        assert not has_warn_tagged(r, "A1")

    def test_png_warns_for_visa(self, tmp_path):
        path = make_png(tmp_path, 1200, 1600)
        r = CheckResult("test")
        append_tier_a(r, path, "visa", PROFILES["visa"])
        assert not has_fail_tagged(r, "A1"), "PNG should not FAIL for visa, only WARN"
        assert has_warn_tagged(r, "A1"), "PNG should WARN for visa"

    def test_png_does_not_warn_for_passport(self, tmp_path):
        path = make_png(tmp_path, 827, 1063)
        r = CheckResult("test")
        append_tier_a(r, path, "passport", PROFILES["passport"])
        assert not has_warn_tagged(r, "A1"), "PNG should not warn for passport"
        assert not has_fail_tagged(r, "A1")

    def test_unsupported_format_fails(self, tmp_path):
        path = tmp_path / "img.bmp"
        arr = np.full((531, 413, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(path), arr)
        r = CheckResult("test")
        append_tier_a(r, path, "passport", PROFILES["passport"])
        assert has_fail_tagged(r, "A1")


class TestTierA_FileSize:
    def test_file_below_minimum_fails(self, tmp_path):
        # Tiny image guaranteed < 1 KB
        path = make_jpeg(tmp_path, 5, 5)
        profile = {**PROFILES["passport"], "file_size_min_kb": 5}
        r = CheckResult("test")
        append_tier_a(r, path, "passport", profile)
        assert has_fail_tagged(r, "A2"), f"Tiny JPEG should fail A2; findings: {r.findings}"

    def test_file_above_maximum_fails(self, tmp_path):
        path = make_noisy_jpeg(tmp_path, 100, 100)
        profile = {**PROFILES["passport"], "file_size_max_kb": 1}
        r = CheckResult("test")
        append_tier_a(r, path, "passport", profile)
        assert has_fail_tagged(r, "A2"), f"Should FAIL A2 above 1 KB; findings: {r.findings}"

    def test_file_in_range_passes(self, tmp_path):
        path = make_noisy_jpeg(tmp_path, 50, 50)
        profile = {**PROFILES["passport"], "file_size_min_kb": 0, "file_size_max_kb": 99999}
        r = CheckResult("test")
        append_tier_a(r, path, "passport", profile)
        assert not has_fail_tagged(r, "A2")


class TestTierA_Dimensions:
    def test_below_minimum_fails(self, tmp_path):
        path = make_jpeg(tmp_path, 200, 300)
        r = CheckResult("test")
        append_tier_a(r, path, "passport", PROFILES["passport"])
        assert has_fail_tagged(r, "A3")

    def test_at_minimum_passes(self, tmp_path):
        path = make_jpeg(tmp_path, 413, 531)
        r = CheckResult("test")
        # Zero out file-size limits so only dimension/aspect is tested
        profile = {**PROFILES["passport"], "file_size_min_kb": 0, "file_size_max_kb": 99999}
        append_tier_a(r, path, "passport", profile)
        fails_a3 = [m for m in fails(r) if "A3" in m]
        assert not fails_a3, f"413×531 should pass A3; A3 failures: {fails_a3}"

    def test_print_resolution_warns_below_600dpi(self, tmp_path):
        path = make_jpeg(tmp_path, 413, 531)
        profile = {**PROFILES["passport"], "file_size_min_kb": 0, "file_size_max_kb": 99999}
        r = CheckResult("test")
        append_tier_a(r, path, "passport", profile)
        assert has_warn_tagged(r, "A3"), "413×531 should WARN about print resolution"

    def test_print_resolution_no_warn_at_600dpi(self, tmp_path):
        path = make_jpeg(tmp_path, 827, 1063)
        profile = {**PROFILES["passport"], "file_size_min_kb": 0, "file_size_max_kb": 99999}
        r = CheckResult("test")
        append_tier_a(r, path, "passport", profile)
        warn_a3 = [m for m in warns(r) if "A3" in m and "print" in m.lower()]
        assert not warn_a3, "827×1063 should not warn about print resolution"

    def test_preferred_size_warns_for_visa(self, tmp_path):
        path = make_jpeg(tmp_path, 354, 472)
        profile = {**PROFILES["visa"], "file_size_min_kb": 0, "file_size_max_kb": 99999}
        r = CheckResult("test")
        append_tier_a(r, path, "visa", profile)
        assert has_warn_tagged(r, "A3"), "354×472 for visa should WARN about preferred size"


class TestTierA_AspectRatio:
    def _profile_no_filesize(self, name):
        return {**PROFILES[name], "file_size_min_kb": 0, "file_size_max_kb": 99999}

    def test_passport_35x45_passes(self, tmp_path):
        """Standard 35×45 mm (aspect 0.778) must pass."""
        path = make_jpeg(tmp_path, 413, 531)
        r = CheckResult("test")
        append_tier_a(r, path, "passport", self._profile_no_filesize("passport"))
        assert not has_fail_tagged(r, "A3"), f"35×45 aspect should pass; fails: {fails(r)}"

    def test_passport_40x50_passes(self, tmp_path):
        """40×50 mm (aspect 0.80) was previously rejected — must now pass."""
        # 40:50 = 0.80; use 424×531 (>= minimum 413×531)
        path = make_jpeg(tmp_path, 425, 531)
        r = CheckResult("test")
        append_tier_a(r, path, "passport", self._profile_no_filesize("passport"))
        fails_a3 = [m for m in fails(r) if "A3" in m]
        assert not fails_a3, f"40:50 aspect (425×531) should pass; A3 fails: {fails_a3}"

    def test_passport_too_wide_fails(self, tmp_path):
        # aspect > 40/45 = 0.889: use 700×531 (aspect 1.32)
        path = make_jpeg(tmp_path, 700, 531)
        r = CheckResult("test")
        append_tier_a(r, path, "passport", self._profile_no_filesize("passport"))
        assert has_fail_tagged(r, "A3"), "Wide image should fail A3 aspect"

    def test_passport_too_tall_fails(self, tmp_path):
        # aspect < 35/50 = 0.70: use 413×1000 (aspect 0.413)
        path = make_jpeg(tmp_path, 413, 1000)
        r = CheckResult("test")
        append_tier_a(r, path, "passport", self._profile_no_filesize("passport"))
        assert has_fail_tagged(r, "A3"), "Tall image should fail A3 aspect"

    def test_visa_3x4_passes(self, tmp_path):
        path = make_jpeg(tmp_path, 354, 472)
        r = CheckResult("test")
        append_tier_a(r, path, "visa", self._profile_no_filesize("visa"))
        fails_a3 = [m for m in fails(r) if "A3" in m]
        assert not fails_a3, f"3:4 visa image should pass; A3 fails: {fails_a3}"

    def test_visa_35x45_passes(self, tmp_path):
        """Standard 35×45 mm passport photo (aspect 0.778) must also pass the visa profile."""
        path = make_jpeg(tmp_path, 413, 531)
        r = CheckResult("test")
        append_tier_a(r, path, "visa", self._profile_no_filesize("visa"))
        fails_a3 = [m for m in fails(r) if "A3" in m]
        assert not fails_a3, f"35:45 aspect should pass visa; A3 fails: {fails_a3}"

    def test_visa_square_fails_aspect(self, tmp_path):
        path = make_jpeg(tmp_path, 500, 500)
        r = CheckResult("test")
        append_tier_a(r, path, "visa", self._profile_no_filesize("visa"))
        assert has_fail_tagged(r, "A3"), "Square should fail A3 aspect for visa"


class TestTierA_Colour:
    def test_greyscale_fails(self, tmp_path):
        path = make_grayscale_jpeg(tmp_path, 413, 531)
        r = CheckResult("test")
        append_tier_a(r, path, "passport", PROFILES["passport"])
        assert has_fail_tagged(r, "A4")

    def test_rgb_passes(self, tmp_path):
        path = make_jpeg(tmp_path, 413, 531)
        r = CheckResult("test")
        append_tier_a(r, path, "passport", PROFILES["passport"])
        assert not has_fail_tagged(r, "A4")


# ---------------------------------------------------------------------------
# Tier B
# ---------------------------------------------------------------------------

class TestTierB_Sharpness:
    def test_blurry_uniform_image_fails(self, tmp_path):
        # Uniform image → Laplacian variance = 0
        path = make_jpeg(tmp_path, 200, 200, color=(180, 180, 180))
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        assert has_fail_tagged(r, "B1"), f"Uniform image should fail sharpness; findings: {r.findings}"

    def test_noisy_image_passes(self, tmp_path):
        path = make_noisy_jpeg(tmp_path, 200, 200)
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        assert not has_fail_tagged(r, "B1"), "Noisy image should pass sharpness"

    def test_moderate_sharpness_passes_b1(self, tmp_path):
        """Sinusoidal pattern gives Laplacian variance ~29 — above new threshold (20) but below old (100)."""
        path = make_sine_jpeg(tmp_path, 200, 200, amplitude=20, wavelength=10)
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        assert not has_fail_tagged(r, "B1"), (
            f"Sine-pattern image (Laplacian variance ~29) should pass B1; findings: {r.findings}"
        )


class TestTierB_Brightness:
    def test_dark_image_fails(self, tmp_path):
        # All-black → L well below min 64
        path = make_jpeg(tmp_path, 200, 200, color=(5, 5, 5))
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        assert has_fail_tagged(r, "B2"), "Dark image should fail brightness"

    def test_medium_grey_passes(self, tmp_path):
        # RGB (128,128,128) → OpenCV Lab L ≈ 128, within [64, 230]
        path = make_jpeg(tmp_path, 200, 200, color=(128, 128, 128))
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        assert not has_fail_tagged(r, "B2"), f"Medium grey should pass brightness; finds: {r.findings}"

    def test_overexposed_image_fails(self, tmp_path):
        # Near-white → OpenCV Lab L ≈ 250+ > max 230
        path = make_jpeg(tmp_path, 200, 200, color=(252, 252, 252), quality=99)
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        assert has_fail_tagged(r, "B2"), "Overexposed image should fail brightness"


class TestTierB_Background:
    def test_white_background_passes(self, tmp_path):
        path = make_jpeg(tmp_path, 200, 200, color=(255, 255, 255))
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        assert not has_fail_tagged(r, "B3"), f"White bg should pass B3; findings: {r.findings}"

    def test_light_grey_background_passes_passport(self, tmp_path):
        """L* ~79 (RGB 192,192,192) must now pass — was failing at old L* 90 threshold."""
        path = make_jpeg(tmp_path, 200, 200, color=(192, 192, 192))
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        bg_dark_fails = [m for m in fails(r) if "B3" in m and "too dark" in m]
        assert not bg_dark_fails, f"Light grey should pass B3 background; B3 fails: {bg_dark_fails}"

    def test_coloured_background_fails(self, tmp_path):
        # Green background — high a*b* saturation
        path = make_jpeg(tmp_path, 200, 200, color=(0, 200, 0))
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        assert has_fail_tagged(r, "B3"), "Coloured background should fail B3"

    def test_dark_background_fails(self, tmp_path):
        path = make_jpeg(tmp_path, 200, 200, color=(30, 30, 30))
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        assert has_fail_tagged(r, "B3"), "Dark background should fail B3"

    def test_high_contrast_face_passes_both_profiles(self, tmp_path):
        """White border + dark face centre → contrast check passes for both profiles."""
        path = make_contrast_jpeg(tmp_path, 300, 400, border_gray=240, face_gray=60)
        face_facts = {"n_faces": 1, "face_box_frac": (0.3, 0.3, 0.4, 0.4)}
        for profile_name in ("passport", "visa"):
            r = CheckResult("test")
            append_tier_b(r, path, profile_name, PROFILES[profile_name], face_facts)
            contrast_fails = [m for m in fails(r) if "contrast" in m.lower()]
            assert not contrast_fails, (
                f"{profile_name}: high-contrast face should pass; contrast fails: {contrast_fails}"
            )

    def test_low_contrast_face_fails_both_profiles(self, tmp_path):
        """Light border + near-same-brightness face → contrast check fails for both profiles."""
        path = make_contrast_jpeg(tmp_path, 300, 400, border_gray=235, face_gray=230)
        face_facts = {"n_faces": 1, "face_box_frac": (0.3, 0.3, 0.4, 0.4)}
        for profile_name in ("passport", "visa"):
            r = CheckResult("test")
            append_tier_b(r, path, profile_name, PROFILES[profile_name], face_facts)
            contrast_fails = [m for m in fails(r) if "contrast" in m.lower()]
            assert contrast_fails, (
                f"{profile_name}: low-contrast face should fail; findings: {r.findings}"
            )

    def test_tightly_framed_portrait_passes_b3_uniformity_with_face_facts(self, tmp_path):
        """When face box is provided, B3 samples only the top background strip.

        A correctly framed portrait has hair/clothing at the left/right/bottom edges
        so 4-edge sampling would fail uniformity.  Top-only sampling sees only the
        plain white wall behind the subject.
        """
        path = make_portrait_tight_jpeg(tmp_path, 300, 400)
        face_facts = {"n_faces": 1, "face_box_frac": (0.2, 0.3, 0.6, 0.5)}
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], face_facts)
        b3_uniform_fails = [m for m in fails(r) if "B3" in m and "uniform" in m]
        assert not b3_uniform_fails, (
            "Portrait with white top and dark body should pass B3 uniformity when face_facts given; "
            f"B3 fails: {b3_uniform_fails}"
        )

    def test_tightly_framed_portrait_fails_b3_uniformity_without_face_facts(self, tmp_path):
        """Without face box, fall back to 4-edge sampling — mixed light/dark border fails uniformity."""
        path = make_portrait_tight_jpeg(tmp_path, 300, 400)
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        b3_uniform_fails = [m for m in fails(r) if "B3" in m and "uniform" in m]
        assert b3_uniform_fails, (
            "Portrait with mixed border should fail B3 uniformity without face_facts; "
            f"findings: {r.findings}"
        )

    def test_contrast_check_applied_to_passport(self, tmp_path):
        """Passport profile must now check contrast (was previously visa-only)."""
        # Same image, check passport specifically
        path = make_contrast_jpeg(tmp_path, 300, 400, border_gray=235, face_gray=230)
        face_facts = {"n_faces": 1, "face_box_frac": (0.3, 0.3, 0.4, 0.4)}
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], face_facts)
        contrast_findings = [m for _, m in r.findings if "contrast" in m.lower()]
        assert contrast_findings, "Passport B3 must include a contrast finding"


# ---------------------------------------------------------------------------
# Tier C (face geometry — mock face_facts, no real images needed)
# ---------------------------------------------------------------------------

class TestTierC_FaceCount:
    def test_none_facts_fails_c1(self):
        r = CheckResult("test")
        append_face_geometry(r, None, PROFILES["passport"])
        assert has_fail_tagged(r, "C1")

    def test_no_face_fails_c1(self):
        r = CheckResult("test")
        append_face_geometry(r, {"n_faces": 0}, PROFILES["passport"])
        assert has_fail_tagged(r, "C1")

    def test_two_faces_fails_c1(self):
        r = CheckResult("test")
        append_face_geometry(r, {"n_faces": 2}, PROFILES["passport"])
        assert has_fail_tagged(r, "C1")

    def test_one_face_passes_c1(self):
        r = CheckResult("test")
        facts = {
            "n_faces": 1,
            "height_frac": 0.75 / HEAD_FACE_RATIO,
            "centre_x_frac": 0.5,
            "n_eyes_upper": 2,
        }
        append_face_geometry(r, facts, PROFILES["passport"])
        assert not has_fail_tagged(r, "C1")


class TestTierC_HeadHeight:
    def _facts(self, head_frac: float) -> dict:
        return {
            "n_faces": 1,
            "height_frac": head_frac / HEAD_FACE_RATIO,
            "centre_x_frac": 0.5,
            "n_eyes_upper": 2,
        }

    def test_head_in_band_passes_c2(self):
        # 75% head height — middle of the 70–80% band
        r = CheckResult("test")
        append_face_geometry(r, self._facts(0.75), PROFILES["passport"])
        assert not has_fail_tagged(r, "C2")

    def test_head_too_small_fails_c2(self):
        # 50% head height — below 70%
        r = CheckResult("test")
        append_face_geometry(r, self._facts(0.50), PROFILES["passport"])
        assert has_fail_tagged(r, "C2")

    def test_head_too_large_fails_c2(self):
        # 90% head height — above 80%
        r = CheckResult("test")
        append_face_geometry(r, self._facts(0.90), PROFILES["passport"])
        assert has_fail_tagged(r, "C2")

    def test_head_at_lower_bound_passes_c2(self):
        r = CheckResult("test")
        append_face_geometry(r, self._facts(0.70), PROFILES["passport"])
        assert not has_fail_tagged(r, "C2")

    def test_head_at_upper_bound_passes_c2(self):
        r = CheckResult("test")
        append_face_geometry(r, self._facts(0.80), PROFILES["passport"])
        assert not has_fail_tagged(r, "C2")


class TestTierC_Centring:
    def _facts(self, cx: float) -> dict:
        return {
            "n_faces": 1,
            "height_frac": 0.75 / HEAD_FACE_RATIO,
            "centre_x_frac": cx,
            "n_eyes_upper": 2,
        }

    def test_centred_face_passes_c3(self):
        r = CheckResult("test")
        append_face_geometry(r, self._facts(0.5), PROFILES["passport"])
        assert not has_fail_tagged(r, "C3")

    def test_slightly_off_centre_still_passes(self):
        # ±10% tolerance — 58% should still pass
        r = CheckResult("test")
        append_face_geometry(r, self._facts(0.58), PROFILES["passport"])
        assert not has_fail_tagged(r, "C3")

    def test_face_too_far_left_fails_c3(self):
        r = CheckResult("test")
        append_face_geometry(r, self._facts(0.2), PROFILES["passport"])
        assert has_fail_tagged(r, "C3")

    def test_face_too_far_right_fails_c3(self):
        r = CheckResult("test")
        append_face_geometry(r, self._facts(0.85), PROFILES["passport"])
        assert has_fail_tagged(r, "C3")


class TestTierC_Eyes:
    def _facts(self, n_eyes: int) -> dict:
        return {
            "n_faces": 1,
            "height_frac": 0.75 / HEAD_FACE_RATIO,
            "centre_x_frac": 0.5,
            "n_eyes_upper": n_eyes,
        }

    def test_two_eyes_passes_c4(self):
        r = CheckResult("test")
        append_face_geometry(r, self._facts(2), PROFILES["passport"])
        assert not has_fail_tagged(r, "C4")

    def test_three_eyes_passes_c4(self):
        # > 2 eyes detected (false positives) still passes
        r = CheckResult("test")
        append_face_geometry(r, self._facts(3), PROFILES["passport"])
        assert not has_fail_tagged(r, "C4")

    def test_one_eye_warns_c4(self):
        r = CheckResult("test")
        append_face_geometry(r, self._facts(1), PROFILES["passport"])
        assert not has_fail_tagged(r, "C4"), "One eye should WARN, not FAIL"
        assert has_warn_tagged(r, "C4")

    def test_no_eyes_fails_c4(self):
        r = CheckResult("test")
        append_face_geometry(r, self._facts(0), PROFILES["passport"])
        assert has_fail_tagged(r, "C4")


# ---------------------------------------------------------------------------
# Report formatting and exit-code logic
# ---------------------------------------------------------------------------

class TestPrintReport:
    def _all_pass(self, name: str) -> CheckResult:
        r = CheckResult(f"Profile: {name}")
        r.ok("A1 format JPEG (.jpg) — accepted")
        r.ok("A2 file size 500 KB — within 50–2048 KB")
        r.ok("A3 image 827×1063 px — meets minimum 413×531 px")
        r.ok("A3 aspect ratio 0.778 — within valid range 0.700–0.889")
        r.ok("A4 image is colour (mode RGB)")
        r.ok("B1 sharpness OK (Laplacian variance 44)")
        r.ok("B2 brightness OK (mean L 128/255)")
        r.ok("B3 background OK (L 250/255, saturation 0.0, variance 0)")
        r.ok("C1 exactly one frontal face detected")
        r.ok("C2 estimated head height 75% of frame (target 70%–80%)")
        r.ok("C3 face horizontally centred (centre at 50% of width)")
        r.ok("C4 2 eyes detected in upper half of face")
        return r

    def _with_fail(self, name: str) -> CheckResult:
        r = CheckResult(f"Profile: {name}")
        r.ok("A1 format JPEG (.jpg) — accepted")
        r.fail("A2 file size 10 KB — below minimum 70 KB")
        r.ok("A3 image 827×1063 px — meets minimum 354×472 px")
        r.ok("A3 aspect ratio 0.778 — within valid range 0.700–0.889")
        r.ok("A4 image is colour (mode RGB)")
        r.ok("B1 sharpness OK (Laplacian variance 44)")
        r.ok("B2 brightness OK (mean L 128/255)")
        r.ok("B3 background OK (L 250/255, saturation 0.0, variance 0)")
        r.ok("C1 exactly one frontal face detected")
        r.ok("C2 estimated head height 75% of frame (target 70%–80%)")
        r.ok("C3 face horizontally centred (centre at 50% of width)")
        r.ok("C4 2 eyes detected in upper half of face")
        return r

    # --- return value / exit-code logic ---

    def test_both_pass_returns_true(self, capsys):
        assert print_report([self._all_pass("passport"), self._all_pass("visa")], "x.jpg") is True

    def test_one_fail_returns_false(self, capsys):
        assert print_report([self._all_pass("passport"), self._with_fail("visa")], "x.jpg") is False

    def test_single_pass_returns_true(self, capsys):
        assert print_report([self._all_pass("passport")], "x.jpg") is True

    def test_single_fail_returns_false(self, capsys):
        assert print_report([self._with_fail("passport")], "x.jpg") is False

    # --- side-by-side output (--profile both) ---

    def test_both_profiles_output_names_present(self, capsys):
        print_report([self._all_pass("passport"), self._all_pass("visa")], "x.jpg")
        out = capsys.readouterr().out
        assert "PASSPORT" in out
        assert "VISA" in out

    def test_per_profile_verdict_shown(self, capsys):
        print_report([self._all_pass("passport"), self._with_fail("visa")], "x.jpg")
        out = capsys.readouterr().out
        # The combined summary line must contain both verdicts on the same line.
        summary_line = next(
            (line for line in out.splitlines() if "PASSPORT:" in line and "VISA:" in line), None
        )
        assert summary_line is not None, "No combined summary line found"
        assert "PASSPORT: PASS" in summary_line
        assert "VISA: FAIL" in summary_line

    def test_both_pass_verdict_line(self, capsys):
        print_report([self._all_pass("passport"), self._all_pass("visa")], "x.jpg")
        out = capsys.readouterr().out
        assert "PASSPORT: PASS" in out
        assert "VISA: PASS" in out

    # --- single-profile output unchanged ---

    def test_single_profile_overall_pass_line(self, capsys):
        print_report([self._all_pass("passport")], "x.jpg")
        out = capsys.readouterr().out
        assert "PASS — photo meets all checked requirements." in out

    def test_single_profile_overall_fail_line(self, capsys):
        print_report([self._with_fail("passport")], "x.jpg")
        out = capsys.readouterr().out
        assert "FAIL — one or more requirements not met." in out


# ---------------------------------------------------------------------------
# Re-shoot guidance — B1/B2/B3 failures must include re-take text
# ---------------------------------------------------------------------------

class TestReshootGuidance:
    def test_blur_fail_has_retake_guidance(self, tmp_path):
        path = make_jpeg(tmp_path, 200, 200, color=(180, 180, 180))
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        b1_fails = [m for m in fails(r) if "B1" in m]
        assert b1_fails, "Uniform image should fail B1"
        assert any("re-take" in m.lower() for m in b1_fails), (
            f"B1 fail must include re-take guidance; got: {b1_fails}"
        )

    def test_underexposed_fail_has_retake_guidance(self, tmp_path):
        path = make_jpeg(tmp_path, 200, 200, color=(5, 5, 5))
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        b2_fails = [m for m in fails(r) if "B2" in m]
        assert b2_fails, "Dark image should fail B2"
        assert any("re-take" in m.lower() for m in b2_fails), (
            f"B2 underexposed fail must include re-take guidance; got: {b2_fails}"
        )

    def test_overexposed_fail_has_retake_guidance(self, tmp_path):
        path = make_jpeg(tmp_path, 200, 200, color=(252, 252, 252), quality=99)
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        b2_fails = [m for m in fails(r) if "B2" in m]
        assert b2_fails, "Near-white image should fail B2"
        assert any("re-take" in m.lower() for m in b2_fails), (
            f"B2 overexposed fail must include re-take guidance; got: {b2_fails}"
        )

    def test_background_colour_fail_has_retake_guidance(self, tmp_path):
        path = make_jpeg(tmp_path, 200, 200, color=(0, 200, 0))
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        b3_fails = [m for m in fails(r) if "B3" in m]
        assert b3_fails, "Green background should fail B3"
        assert any("re-take" in m.lower() for m in b3_fails), (
            f"B3 colour fail must include re-take guidance; got: {b3_fails}"
        )

    def test_background_dark_fail_has_retake_guidance(self, tmp_path):
        path = make_jpeg(tmp_path, 200, 200, color=(30, 30, 30))
        r = CheckResult("test")
        append_tier_b(r, path, "passport", PROFILES["passport"], None)
        b3_fails = [m for m in fails(r) if "B3" in m]
        assert b3_fails, "Dark background should fail B3"
        assert any("re-take" in m.lower() for m in b3_fails), (
            f"B3 dark fail must include re-take guidance; got: {b3_fails}"
        )


# ---------------------------------------------------------------------------
# JPEG size iteration — _save_jpeg_within_size and crop_to_profile (visa)
# ---------------------------------------------------------------------------

class TestJpegSizeIteration:
    def test_in_band_quality_95(self, tmp_path):
        """Quality 95 already in band → fast path, file within bounds."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        out = tmp_path / "out.jpg"
        size_kb = _save_jpeg_within_size(img, out, 0, 99999)
        assert out.exists()
        assert 0 <= size_kb  # any size accepted

    def test_caps_at_max_kb(self, tmp_path):
        """Binary search must cap file size at max_kb."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        out = tmp_path / "out.jpg"
        # Set tight max to force the search
        ok, raw = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        q95_kb = len(raw) / 1024
        max_kb = q95_kb * 0.5  # half of quality-95 size
        size_kb = _save_jpeg_within_size(img, out, 0, max_kb)
        assert size_kb <= max_kb, f"Output {size_kb:.1f} KB exceeds max {max_kb:.1f} KB"

    def test_visa_fixed_file_size_in_band(self, tmp_path):
        """Auto-fixed visa JPEG must be within 70 KB – 3.5 MB."""
        src = make_noisy_jpeg(tmp_path, 600, 900)
        face_facts = {
            "n_faces": 1,
            "height_frac": 0.50 / HEAD_FACE_RATIO,
            "centre_x_frac": 0.5,
            "face_box_frac": (0.2, 0.3, 0.6, 0.4),
            "n_eyes_upper": 2,
        }
        out_path = tmp_path / "visa_fixed.jpg"
        assert crop_to_profile(src, PROFILES["visa"], face_facts, out_path)
        size_kb = out_path.stat().st_size / 1024
        min_kb = PROFILES["visa"]["file_size_min_kb"]
        max_kb = PROFILES["visa"]["file_size_max_kb"]
        assert size_kb >= min_kb, f"Fixed visa JPEG {size_kb:.0f} KB below min {min_kb} KB"
        assert size_kb <= max_kb, f"Fixed visa JPEG {size_kb:.0f} KB above max {max_kb} KB"

    def test_visa_fixed_dimensions(self, tmp_path):
        """Auto-fixed visa image must be exactly fix_width_px × fix_height_px."""
        from PIL import Image as PILImage
        src = make_noisy_jpeg(tmp_path, 600, 900, name="src2.jpg")
        face_facts = {
            "n_faces": 1,
            "height_frac": 0.50 / HEAD_FACE_RATIO,
            "centre_x_frac": 0.5,
            "face_box_frac": (0.2, 0.3, 0.6, 0.4),
            "n_eyes_upper": 2,
        }
        out_path = tmp_path / "visa_dims.jpg"
        assert crop_to_profile(src, PROFILES["visa"], face_facts, out_path)
        with PILImage.open(out_path) as img:
            w, h = img.size
        assert w == PROFILES["visa"]["fix_width_px"]
        assert h == PROFILES["visa"]["fix_height_px"]
