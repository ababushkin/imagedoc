# imagedoc

Local, offline command-line tools for preparing identity-document images:

- **`convert.py`** — batch-converts JPG images to PDFs.
- **`check.py`** — checks a photo against Australian passport and visa photo specs, and can auto-fix framing.

Everything runs on your machine; no image is ever uploaded. The Australian Passport Office warns against online photo services for exactly this reason — sending your face to a third party is an identity-fraud risk. `check.py` is **advisory**: it catches the common, mechanically-checkable failures before you print or upload, but the Passport Office and Home Affairs make the final call.

## Document requirements

`check.py` validates against two profiles. Pick one with `--profile`, or check both at once (the default). The figures below are verified against the official Australian government guidance (last checked May 2026):

- **Passport** — [Australian Passport Office, *Passport photo guidelines*](https://www.passports.gov.au/passport-photo-guidelines)
- **Visa / citizenship** — [Department of Home Affairs, *Photo requirements for citizenship applications*](https://immi.homeaffairs.gov.au/citizenship/photo-requirements-for-citizenship-applications), the same photo standard Home Affairs applies to ImmiAccount uploads

Both standards describe the *same physical photo*; they differ only in how it is submitted — a glossy print for passports, a digital file for online lodgement.

| Requirement | Official rule | Profile |
|---|---|---|
| Photo size | 35–40 mm wide × 45–50 mm high (aspect ratio 0.70–0.89) | both |
| Face height (chin to crown) | 32–36 mm | both |
| Background | plain white or light (L* ≥ 75, i.e. no darker than light grey), contrasting with the face | both |
| Recency, expression, etc. | < 6 months old, colour, neutral expression, no glasses (medical exception aside), no retouching, even lighting | both |
| File format | JPEG | `visa` |
| File size | 70 KB – 3.5 MB | `visa` |
| Resolution | preferred 1200 × 1600 px | `visa` |

The printed-passport spec is stated only in millimetres and on paper terms (dye-sublimation, heavy-weight glossy, ≥ 200 gsm) — there is **no official DPI or minimum-pixel rule**. So `check.py` adds its own pixel floors to judge a digital file, and these are tool choices, not government numbers:

- **`passport`** — accepts 413 × 531 px (35 × 45 mm at 300 DPI) as the minimum; warns below 827 × 1063 px (600 DPI) since that prints sharper on glossy paper.
- **`visa`** — fails below 354 × 472 px (the smallest 3:4 frame it trusts) and treats 1200 × 1600 px as the preferred size.

One deliberate deviation from the letter of the spec: both profiles accept PNG as well as JPEG. Home Affairs asks for JPEG only; PNG is accepted but triggers a warning for the `visa` profile since the ImmiAccount portal may reject it.

Beyond the table, both profiles check image quality and face geometry: in-focus (not blurry), correctly exposed, a uniform background, exactly one frontal face, the face horizontally centred, and both eyes visible.

**What it does not check.** It does not verify the photo is recent, unedited, or a biometric match — those aren't derivable from pixels, and the government runs its own match. It never alters image content: a wrong background colour, blur, or bad exposure is reported as a FAIL with advice to re-shoot, never silently "fixed."

> **Note on head height.** The head-height check measures the detected face box (brow to chin), which reads lower than the full chin-to-crown head height the specs define. A raw, uncropped portrait is expected to fall short here; the band is the target *after* cropping.

## Usage

### Converting JPGs to PDFs

1. Place your `.jpg` / `.jpeg` images into the `input/` folder.
2. Run the script:
   ```
   python3 convert.py
   ```
3. Find the converted PDFs in the `output/` folder.

### Checking a photo

Check a photo against the chosen profile(s):

```
python3 check.py <image> [--profile passport|visa|both]
```

The command prints a table of PASS/FAIL results for each checked requirement. It exits with code 0 if all checks pass, or code 1 if any check fails — so you can use it in a script as a gate.

**Profiles:**
- `passport` — physical photo specifications for Australian passport applications (35–40 mm × 45–50 mm, 300+ DPI recommended)
- `visa` — digital upload specifications for Australian visa and citizenship applications (matches the same physical photo standard, but with pixel-size and file-size constraints for online lodgement)
- `both` (default) — checks against both profiles and reports them side by side

### Auto-fixing a photo

If the photo has geometry or framing issues, you can auto-fix the most common problems:

```
python3 check.py <image> [--profile passport|visa|both] --fix
```

or write the fixed image to a specific location:

```
python3 check.py <image> [--profile passport|visa|both] -o fixed.jpg
```

The `--fix` option (or `--o`, which implies `--fix`) will:
1. Crop the image to reframe the face to the target head height (70–80% of the image height).
2. Scale to the preferred dimensions for the chosen profile.
3. Re-encode as JPEG.
4. Re-run checks on the result to confirm it passes.

Fixing is **geometric only** — it crops and scales, but never retouches the image, changes colours, or alters the background. If the background is the wrong colour, the exposure is wrong, or the face is out of focus, fixing cannot address those — re-shoot the photo with the correct settings.

### Non-claims — what check.py does not verify

`check.py` is **advisory only**. It catches common, mechanically-detectable failures before you print or upload, but the Australian Passport Office and Department of Home Affairs make the final call on acceptance.

The tool does **not** verify:
- **Recency** — it cannot confirm the photo is < 6 months old.
- **Unedited** — it cannot detect retouching or digital edits.
- **Background replacement** — it does not detect if the background has been artificially added or removed.
- **Biometric match** — it is not a substitute for official identity verification.

The Passport Office and Home Affairs will conduct their own checks, including recency and authenticity, when you lodge your application. A photo that passes `check.py` is compliant with the format and quality specs, but not guaranteed to be accepted.

## Installation

The toolchain is pinned for reproducibility:

- **Python version** — `.mise.toml` pins 3.14.4, managed by [mise](https://mise.jdx.dev).
- **Dependencies** — `requirements.txt` pins exact versions: Pillow (PDF conversion), plus NumPy and OpenCV (the photo checker).

One-time setup:

```
mise install                              # installs the pinned Python (3.14.4)
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

After that, run the scripts directly — they add `.venv` to the path themselves, so no manual activation is needed.

Not using mise? Install Python 3.14.x yourself, then run the `venv` and `pip install` steps above.
