# Per_Platform_aug — Standalone branch

This branch contains a **CSV-free runtime** implementation of measurement-derived Facebook, Instagram and KakaoTalk image/video augmentation. The published source is an experimental approximation, not a verified platform encoder clone.

- [Standalone Python preprocessing script](./platform_aware_augment_standalone.py)
- [Korean usage guide and measurement caveats](./README_standalone_KR.md)
- [Source provenance (SHA-256 manifest)](./standalone_policy_provenance.json)

Quick start:

```bash
pip install -r requirements.txt
python platform_aware_augment_standalone.py --show-profiles
python platform_aware_augment_standalone.py --input /path/to/train --output /path/to/augmented --copies 3 --seed 42
```

Video processing requires FFmpeg and FFprobe (including libx264/libx265). This branch **does not require the 16 measurement CSVs at runtime**. Original measurement CSVs and Prosecutor's Office datasets are not uploaded to this public repository. Validate generated compression, SSIM, brightness and contrast distributions offline before model training.

The original CSV-dependent implementation remains on [main](https://github.com/kimmyju/Per_Platform_aug/tree/main).
