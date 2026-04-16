# Asset Generation Notes

This file documents how the binary assets under
`web/ecf8/assets/figures/` were produced, so they can be regenerated
from raw outputs if needed.

All commands assume:
- macOS with Homebrew
- `ffmpeg` installed (`brew install ffmpeg`)
- `sips` (ships with macOS) for PNG resizing
- Working directory: `web/ecf8/assets/figures/`

---

## 1. Interactive demo videos

The "Interactive Demo" section in `index.html` plays one short video per
sample instead of streaming 30 PNG intermediate frames. Each video is
**1 second long, 30 frames @ 30 fps, 512×512**. Playback speed is
controlled in JS via `video.playbackRate`, so the two panels can finish
at different wall-clock times using the same source file.

### Source
Raw denoising intermediates used to live in:

```
seed2025-original/intermediates_<i>/prompt<i>_000.png ... prompt<i>_029.png
```

These 150 PNGs (~290 MB total) have been deleted from the repo. If you
need to rebuild the demo videos, restore them from Git history (before
commit `3ee62a5`) or regenerate from inference scripts.

### Commands

```bash
cd web/ecf8/assets/figures/seed2025-original

for i in 0 1 2 3 4; do
  # VP9 / WebM — smaller where supported
  ffmpeg -y -framerate 30 -i intermediates_${i}/prompt${i}_%03d.png \
    -vf "scale=512:512:flags=lanczos" \
    -c:v libvpx-vp9 -crf 32 -b:v 0 -pix_fmt yuv420p -an \
    sample_${i}.webm

  # H.264 / MP4 — universal fallback (and actually smaller for our inputs)
  ffmpeg -y -framerate 30 -i intermediates_${i}/prompt${i}_%03d.png \
    -vf "scale=512:512:flags=lanczos" \
    -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p \
    -movflags +faststart -an \
    sample_${i}.mp4
done
```

### Resulting files
`seed2025-original/sample_{0..4}.{mp4,webm}` — roughly 300–520 KB each.
Total demo payload: ~4 MB (down from ~290 MB).

### JS playback timing
In `index.html` the two panels use the same video source but different
playback rates so that:

- ECF8 panel finishes in `T_ECF8 / SPD = 13 / 4 = 3.25 s` wall time
- FP8 panel finishes in `T_FP8  / SPD = 24 / 4 = 6.00 s` wall time

Given the video is `VIDEO_DUR = 1.0 s` long internally:

```
playbackRate = VIDEO_DUR / wallTime
             = 1 / 3.25 ≈ 0.308   (ECF8)
             = 1 / 6.00 ≈ 0.167   (FP8)
```

---

## 2. Qwen-Image gallery

The 10 images in the "Perfectly Lossless Compression" section
(`qwen_image_0.png` ... `qwen_image_9.png`) are rendered in a 5×2 CSS
grid with each tile at roughly 165 CSS px. They previously shipped at
1328×1328 (~2.2 MB each); we downscaled them to 768×768 so they still
look sharp in the lightbox without wasting bandwidth.

### Command

```bash
cd web/ecf8/assets/figures

for i in 0 1 2 3 4 5 6 7 8 9; do
  sips -Z 768 qwen_image_${i}.png --out qwen_image_${i}.png
done
```

`sips -Z 768` resamples so that the longer edge is 768 px, preserving
aspect ratio. Inputs here are square so output is 768×768.

### Resulting sizes
~0.6–1.0 MB per image, ~8.8 MB total (down from ~22 MB).

---

## 3. General conventions

- All binary assets live under `web/ecf8/assets/figures/`.
- PNGs for static figures (entropy plots, lookup table diagram) stay at
  their original resolutions since they contain fine detail.
- Photographic / generative outputs should be downscaled to a
  screen-appropriate size before committing.
- Prefer short muted videos (`<video muted playsinline>`) over
  image-sequence animations for anything with more than a few frames.
