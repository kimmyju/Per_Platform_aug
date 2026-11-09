# main_v2 — Standalone v2 Offline + per-output metadata

This branch keeps the original **`platform_aware_augment_standalone.py` offline
augmentation design** (360 measurement-derived combinations, seed per relative
input path/copy/epoch, calibrated brightness/contrast, video encoding).

**New:** use optional *input metadata* (CSV/JSONL/JSON) and save complete
per-augmentation *output metadata* to a user-defined CSV or JSONL location.
The original measurement CSVs are still **not needed** at runtime.

## Typical use

```bash
git clone -b main_v2 https://github.com/kimmyju/Per_Platform_aug.git
cd Per_Platform_aug
pip install -r requirements.txt
# FFmpeg and FFprobe also required for video; libx264/libx265 support.

python platform_aware_augment_standalone.py \
  --input /media/NAS/test_images \
  --output /local_ssd/augmented \
  --input-metadata /media/NAS/labels/test.csv \
  --input-metadata-key filepath \
  --metadata-match-mode relative_path \
  --metadata-output /media/NAS/results/test_aug_metadata.csv \
  --copies 1 --seed 42 --epoch 0
```

The input metadata path and output metadata path are independent and can
refer to any accessible directories. Output media location is still controlled
by `--output` separately.

If no `--metadata-output` is specified, the result is written to
`OUTPUT/augmentation_manifest.jsonl` (existing default behavior).
If no `--input-metadata` is specified, files are discovered recursively
under `--input` as before and their metadata is stored as `null`.

### Input CSV format

```csv
filepath,id,label,source
folder/picture.jpg,1001,fake,ffpp
folder/another.png,1002,real,celeb
```

The key field is selected with `--input-metadata-key`, which defaults to
`source_relative_path`. The default match mode is `relative_path`:
a relative path is matched against the source filename relative to `--input`.
`--metadata-match-mode basename` matches full filename only, and `stem`
matches filename without extension. Both shorthand modes are rejected if
two source inputs share the same matching name (avoid ambiguous labels).

Input JSON may be a list of records or `{"records": [...]}`. Input JSONL
must contain one JSON object per line.

By default every input media file must match exactly one metadata entry.
If a metadata row is missing, the command stops *before generating outputs*.
Use `--allow-missing-input-metadata` only when unmatched files are intentional.
Duplicate metadata keys fail. Extra unused metadata rows are allowed.

### What is saved for each augmented output?

The output file (.jsonl or .csv) contains one entry per generated copy, not
one entry per source file.

- Source path, source relative path, and generated output path
- Input metadata object/columns, unchanged, under `input_metadata`
- Seed, derived per-file seed, epoch, copy index and execution settings
- Selected platform, quality, band, source measurement row ordinal
- Complete selected reference tuple under `measured_recipe`
- Actual transformation info under `transformation`, including image
  crop/resize/JPEG quality/subsampling and brightness/contrast calibration,
  or video codec, FPS, CRF/target bitrate, GOP, filters, output probing and
  calibration, when available
- Execution status (`ok`, `error`, or `dry_run`)

JSONL preserves all nested metadata. CSV exposes important fields as columns,
while `record_json` stores the entire metadata entry for lossless access.

If `--continue-on-error` is specified, failure records with an error message
are also written. If conversion fails without that flag, the failing record
is written before exiting.

### Determinism

```python
SHA256(f"{seed}|{source_relative_path}|{copy_index}|{epoch}")[:8]
```

Sampling and jitter use this local random generator. Stable input relative
paths, unchanged settings, epoch and copy number select the same recipe
irrespective of media enumeration order. For fair model comparisons, freeze
CSV/inputs, `seed`, `epoch`, `copies`, platform distributions, and code
revision. FFmpeg/Pillow versions can affect encoded pixels. Do not assume
byte-identical outputs across different encoding environments.

**This branch is Offline-only**: all augmented images/videos are saved under
`--output`. If NAS is full, put `--output` on local SSD and delete the media
once inference is complete; metadata may be kept separately on NAS.

### Verification

```bash
python -m unittest -v test_standalone_empirical.py test_standalone_metadata.py
python -m py_compile platform_aware_augment_standalone.py standalone_metadata.py
```

This testing scope covers image conversion, deterministic sampled recipes,
metadata input/output formats, missing and duplicate keys, and failures.
It does not by itself establish exact agreement with actual SNS uploads.

## Nearest measured-profile fallback (new)

By default `--fallback-policy nearest` keeps the original exact matching
when a measured band/orientation and sufficiently close aspect ratio exist.
If not, it selects the nearest **observed joint recipe**, always within the
same platform, modality and quality:

1. Exact band + orientation + log-aspect tolerance (<= 0.15): unchanged sampler.
2. Same band and orientation but no close aspect: nearest observed aspect.
3. No matching orientation in the band: nearest aspect among that band.
4. No measurements in the band: nearest aspect and long-side log size
   across measured bands of that platform/quality.

Never infer missing recipe parameters independently. The selected recipe is
an empirical joint observation, but applying it outside its source geometry
is an **extrapolation**, not a directly observed platform behavior.

For the old strict behavior, pass `--fallback-policy strict`.

The per-output CSV/JSONL now includes these keys:

- `requested_condition`: e.g. `le512/land`
- `selected_condition`: e.g. `le512/port`
- `selection_method`: `exact`, `nearest_aspect`,
  `nearest_same_band` or `nearest_other_band`
- `is_extrapolated`: true for every non-exact selection
- `aspect_log_distance`, `size_log_distance`,
  `selection_candidates`

The existing per-file SHA-256 seed and RNG ordering are preserved for
exact-match inputs. Fallback is deterministic for a fixed input path, seed,
copy index, epoch, policy and software environment. `metadata-output` may
point to any accessible CSV/JSONL path; no extra NAS image cache is required.

Warning: cross-orientation fallback can yield imperfect geometry, especially
for Instagram. Review the generated image and analyze extrapolated cases
separately if they are used in scientific performance claims.
