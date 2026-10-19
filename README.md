# Platform-Aware Augmentation — Empirical Standalone v2

**CSV files are not required at runtime.** The Python script embeds 360 *individual measured joint transformation tuples*, grouped into 12 platform/modality/quality profiles and 65 resolution/orientation cells. Instagram high/mid/low duplicate output conditions are represented as one profile per modality.

## Files
- `platform_aware_augment_standalone.py` — preprocessing CLI
- `README_standalone_KR.md` — protocol and research limitations
- `test_standalone_empirical.py` — policy integrity tests
- `standalone_policy_provenance.json` — SHA-256 of source CSVs, without sharing the CSV data

## Run

```bash
pip install -r requirements.txt
python platform_aware_augment_standalone.py --show-profiles
python platform_aware_augment_standalone.py --input /path/to/train --output /path/to/augmented --copies 3 --seed 42
python test_standalone_empirical.py
```

Image processing requires Pillow and NumPy. Video requires FFmpeg/FFprobe with libx264 and libx265.

## Sampling and photometry
1. Training mixture (proposal): Clean/FB/IG/KT = 25% each.
2. Conditionally choose quality, resolution band and orientation.
3. Randomly sample an **observed tuple** with matched approximate aspect ratio; correlated compression, geometry, temporal and photometric parameters remain together.
4. Brightness/contrast correction is **enabled by default** (`--p-photo 1.0`). First encode without added photometric correction and measure the change. Try residual correction and only retain it if the estimated error decreases.
5. Save operations and residuals in `augmentation_manifest.jsonl`.

Jitter in QF/CRF/bitrate/crop/photo is 0 by default. The 25% mixture, aspect thresholds, interpolation, crop centering and encoder choices are **design decisions, not measured population probabilities**.

**IMPORTANT**: Video photometric fitting uses an FFmpeg 8-bit gray approximation, not the forensic CSV displayed-Y estimator. Generated compression/PSNR/SSIM/b/a remain unverified against the full platform redownload distributions. Do not train on the 480 evaluation samples. The public repo does not contain original measurement CSVs.
