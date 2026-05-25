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
| Photo size | 35–40 mm wide × 45–50 mm high | both |
| Face height (chin to crown) | 32–36 mm | both |
| Background | plain white or light, contrasting with the face (Home Affairs: neutral / light grey) | both |
| Recency, expression, etc. | < 6 months old, colour, neutral expression, no glasses (medical exception aside), no retouching, even lighting | both |
| File format | JPEG | `visa` |
| File size | 70 KB – 3.5 MB | `visa` |
| Resolution | preferred 1200 × 1600 px | `visa` |

The printed-passport spec is stated only in millimetres and on paper terms (dye-sublimation, heavy-weight glossy, ≥ 200 gsm) — there is **no official DPI or minimum-pixel rule**. So `check.py` adds its own pixel floors to judge a digital file, and these are tool choices, not government numbers:

- **`passport`** — accepts 35 × 45 mm at 300 DPI (413 × 531 px) as the minimum; 600 DPI (827 × 1063 px) is sharper and safer for print.
- **`visa`** — fails below 354 × 472 px (the smallest 3:4 frame it trusts) and treats 1200 × 1600 px as the preferred size.

Two deliberate deviations from the letter of the spec: the `passport` profile requires a white background (the office also allows a light contrasting one), and both profiles accept PNG as well as JPEG (Home Affairs asks for JPEG).

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

```
python3 check.py <image> [--profile passport|visa|both] [--fix] [-o out.jpg]
```

- `--profile` — which spec to check against; defaults to `both`.
- `--fix` (or `-o`) — crop, scale, and re-encode the photo so its framing, dimensions, and file size meet the chosen profile, then re-check the result. Fixing is geometric only — it never retouches the image or changes the background.
- `-o` — where to write the fixed image (implies `--fix`).

The command prints a per-rule PASS/FAIL report and exits non-zero if any check fails, so it can gate a script.

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
