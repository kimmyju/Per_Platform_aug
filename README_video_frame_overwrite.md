# Video-frame postprocess: replace selected Perturb images in place

This standalone tool is for an **already completed** image augmentation run.
It does **NOT** re-augment an existing Perturb JPEG. It reads the **original**
frame path from `frames.csv`, uses embedded *video* measurement profiles and
applies a single-I-frame H.264/HEVC codec approximation. It then replaces the
matching **existing** Perturb image file at the exact same path (same filename
and extension) and replaces `Perturb_metadata.csv` with updated rows.

Neither `frames.csv` nor original source frames are modified. The original
`platform_aware_augment_standalone.py` is not modified by this workflow.

## Prerequisites

- `ffmpeg`, `ffprobe`, FFmpeg H.264/H.265 encoders `libx264` and `libx265`
- `pip install -r requirements.txt`
- A completed `Perturb_metadata.csv` whose `output` paths exist and are
  within the specified `--images-root`; input `frames.csv` with `relpath`
  and `filepath` pointing at original images.

## 1. Preview without modifying existing images or metadata

```bash
python postprocess_video_frame.py \
  --frames-csv /media/NAS/DATASET/ProsecutorOffice/Perturb_Test/frames.csv \
  --metadata-csv /media/NAS/DATASET/ProsecutorOffice/Perturb_Test/Perturb_metadata.csv \
  --images-root /media/NAS/DATASET/ProsecutorOffice/Perturb_Test/images \
  --backup-dir /media/NAS/DATASET/ProsecutorOffice/Perturb_Test/backup_video_frame_001 \
  --replace-fraction 0.333333 \
  --seed 42 \
  --plan-only
```

Writes `Perturb_metadata_video_frame_plan.csv` listing original inputs, target
output paths, and previous/new quality labels. The plan file itself must be
a **new** path. It does not overwrite images or the existing metadata CSV.

## 2. Replace files and metadata in place (backup mandatory)

```bash
python postprocess_video_frame.py \
  --frames-csv /media/NAS/DATASET/ProsecutorOffice/Perturb_Test/frames.csv \
  --metadata-csv /media/NAS/DATASET/ProsecutorOffice/Perturb_Test/Perturb_metadata.csv \
  --images-root /media/NAS/DATASET/ProsecutorOffice/Perturb_Test/images \
  --backup-dir /media/NAS/DATASET/ProsecutorOffice/Perturb_Test/backup_video_frame_001 \
  --replace-fraction 0.333333 \
  --seed 42 \
  --overwrite-existing-outputs
```

The backup directory must **not exist yet**. It receives original target
images in an `images/` subdirectory and the original metadata file in
`metadata/Perturb_metadata.csv`. The original output paths are unchanged.
On a normal in-process exception, the tool attempts to restore replaced
outputs and original metadata from these backups. An abrupt kill/power loss
may require manual recovery. **Never rerun the same input with an old backup
directory**, and keep the backups until validation is complete.

## Selection and scientific meaning

- Eligible: successful non-clean image augmentation rows. Clean rows untouched.
- Deterministic fraction selected from eligible rows, stratified approximately
  by class, category, and platform. It preserves the previously selected
  platform, not the old image-specific quality.
- Video quality: Facebook high/mid/low, Instagram shared, KakaoTalk single.
  KakaoTalk `single` refers to a **30-row empirical video profile**, not no
  compression; it includes H.264 (720p) and HEVC (1080p/2160p).
- Selection and measurement fallback stay inside platform/video/quality.
- H.264: measured x264-equivalent CRF per selected video measurement.
- HEVC: default x265 CRF 28 (`--hevc-crf` can change this), which is **an
  explicit engineering approximation, not a measured x265 CRF** because
  the original video bitrate is not available from individual JPEG frames.
- FFmpeg encodes only one I-frame and decodes it. No temporal GOP, B-frame,
  motion prediction, FPS or frame offset effects are reproduced.
- **Existing .jpg filenames stay .jpg**. To avoid broken filename extensions,
  a final JPEG QF100 / 4:4:4 re-save follows video decoding. This is an
  additional lossy delivery step, recorded as such in metadata. Existing
  .png outputs are saved losslessly.
- The transformed pixels derive from ORIGINAL `filepath`, **not** the old
  Perturb output. The source .jpg can already include its own original
  JPEG compression.
- `video_codec`, `h264_crf`, `hevc_crf`,
  `codec_quality_source`, `crop_box`, `resize_to`, `brightness_b`,
  `contrast_a`, `delivery_format`, `delivery_jpeg_qf`,
  `delivery_jpeg_subsampling`, `measured_recipe`,
  `transformation`, `record_json` are updated. The old recipe/settings
  are stored in `previous_augmentation` within `record_json`.
- Input model probability scores in `input_metadata` are **original-frame
  scores**, not scores recalculated on newly compressed images.

See `test_video_frame_postprocess.py` for an end-to-end FFmpeg regression.
