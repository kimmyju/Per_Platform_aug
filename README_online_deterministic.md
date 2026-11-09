# Deterministic online augmentation for Standalone v2

This module is an **additive implementation** based on the existing
`platform_aware_augment_standalone.py`. It calls the original
`MeasurementProfiles.choose`, `transform_image` and `transform_video`
functions directly. The original offline script is unchanged.

## Evaluation design: A and B see the same augmented sample

1. Keep a unique, stable `id` per CSV sample. The ID must be **the same**
   across model A and model B. Do not use CSV row number, DataLoader batch
   index, worker ID, current time or absolute machine path as the ID.
2. Freeze `seed=42`, `epoch=0`, `copy_index=0`, branch/commit, platform
   probabilities, all jitter settings and all model-independent preprocessing.
3. A per-item RNG is seeded as
   `SHA256(f"{seed}|{normalized_key}|{copy_index}|{epoch}")[:8]`,
   following the existing standalone implementation.
4. Each sample is independently augmented; CSV shuffle, DataLoader worker
   order, and interleaving unrelated samples cannot consume its random stream.
5. Image pixels are returned as PIL images; temporary JPEG/mask files are
   removed before returning. Video paths are valid only **inside the context
   manager**, and temporary videos are deleted on exit.
6. Both models should run in the **same software/FFmpeg environment** for
   consistent codec output. A fixed recipe does **not** guarantee bitwise
   identical encodes across different encoder builds and hardware.
7. The output manifest/metadata is small. Keep a JSONL record of the per-item
   decisions during evaluation for auditing. The caller controls logging.

## CSV example

```csv
id,path,label
image-001,images/001.jpg,real
image-002,images/002.jpg,fake
```

```python
from standalone_online import (
    CSVOnlineImageDataset, OnlineConfig, StandaloneOnlineAugmentor,
)

cfg = OnlineConfig(seed=42, epoch=0, temp_dir="/local_ssd/tmp")
aug = StandaloneOnlineAugmentor(cfg)
dataset = CSVOnlineImageDataset(
    "eval.csv", "/media/NAS/test", aug, id_column="id",
    path_column="path", label_column="label", return_info=True
)

# Consume the image in memory, without a persistent augmented test set.
sample = dataset[0]
image = sample["image"]               # PIL.Image
metadata = sample["augmentation"]     # JSON-serializable recipe + transform
# Model-specific preprocessing (resize/normalize/tensor) comes AFTER this step.
```

For a PyTorch DataLoader use an image-to-tensor transform before default
collation, and avoid persistent worker epoch changes unless you propagate the
new config to all worker replicas. For **fixed evaluation**, keep epoch=0.

### Videos

```python
with aug.augment_video(
    "/media/NAS/test/video001.mp4",
    key="video-001", copy_index=0,
) as (augmented_path, metadata):
    # Open, decode, and finish inference INSIDE this block.
    model_input = decode_video(augmented_path)
    predictions = evaluate(model_input)
# Temporary output no longer exists here.
```

`decode_video` and `evaluate` above are placeholders for your existing
evaluation code. For fair A/B comparisons, it is best to decode once and pass
the same image tensor or video frames to both models if possible.

### Force a specific platform/quality

```python
image, mask, info = aug.augment_image(
    "/media/NAS/test/001.jpg", key="image-001",
    force_platform="facebook", force_quality="low", return_info=True
)
```

For a **complete per-platform robustness benchmark**, use a fixed policy
grid instead of relying only on the 25% mixture. Keep consistent IDs and
record platform, quality, selected row ordinal, seed, and config fingerprint.

## Important limitations

- The online image method uses LOCAL short-lived files under `temp_dir`
  to call the **exact standalone image transformation**. This is on-demand
  rather than purely RAM-only, and does not fill your NAS with augmentations.
  It is not a new approximate one-pass implementation.
- Video encoding and Standalone photometric calibration can be expensive.
- Same seed alone cannot reproduce results if input bytes, IDs, code revision,
  dependencies, FFmpeg, or augmentation settings have changed.
- A manifest entry can be reproduced only when these dependencies are frozen.
- Different CSV entries with duplicate IDs are rejected by the CSV wrapper.
- Unsupported resolutions, orientations, or aspect ratios still raise
  errors as the original Standalone implementation does.
- Video mask temporal alignment is not implemented.
- This code does not claim exact recreation of social platforms' encoders.

## Verification

```bash
python -m unittest -v test_standalone_empirical.py test_standalone_online.py
python -m py_compile platform_aware_augment_standalone.py standalone_online.py
```

In addition to unit tests, compare decoded images/frames from two independent
runs with the actual FFmpeg version/environment used for the evaluation.
