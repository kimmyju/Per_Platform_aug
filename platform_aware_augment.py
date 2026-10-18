#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measurement-driven platform-aware augmentation for deepfake robustness.

Reads the 16 per-pair CSVs provided with the ProsecutorOffice robustness study.
The reference transformation is UPLOADED stimulus -> REDOWNLOADED media,
NOT an original master -> redownloaded media transformation.

INSTALL
  python -m pip install Pillow
  # ffmpeg and ffprobe with libx264 + libx265 are needed for videos.

EXAMPLE (offline preprocessor)
  python platform_aware_augment.py \
     --input /path/to/uploaded_training_media \
     --output /path/to/train_augmented \
     --measurements-dir /path/to/folder_containing_16_csvs \
     --copies 3 --seed 42

EXAMPLE (image plus optional segmentation masks):
  python platform_aware_augment.py --input train/images --output augmented/images \
     --measurements-dir ./measurements --mask-root train/masks \
     --mask-output augmented/masks --copies 2

EXAMPLE (inspect dataset-derived policy, no media required):
  python platform_aware_augment.py --measurements-dir ./measurements --show-profiles

GLOBAL ROUTE MIX (proposal, NOT measured platform prevalence):
   clean 25%, Facebook 25%, Instagram 25%, KakaoTalk 25%.
Facebook image qualities: high/mid/low each 1/3.
Facebook video: high 50%, pooled mid+low 50% (mid/low near-duplicates).
Instagram image/video: single deduplicated high profile (high/mid/low match
  measured statistics; *this code alone cannot establish file SHA-256 equality*).
KakaoTalk image: original/normal/low each 1/3.
KakaoTalk video: only one video set, never three invented quality levels.

CONDITIONAL MEASURED PROBABILITIES:
  Given platform + modality + quality + input resolution band, sample a
  row from near-matching input ASPECT RATIO. This preserves empirical
  conditional frequencies in the aspect-matched subset (up to 10 per band).
  Thus e.g. Facebook low image on le512 is no-op 10/10; on le1080 is
  10/10 resize+reencode. These are sample-based experiment probabilities,
  NOT app-level prevalence estimates.

PER-TRANSFORM SETTINGS:
  Geometry: use selected reference's down/orig dimension ratios.
  JPEG: sample measured down_estimated_jpeg_qf with optional +/-1 proposed
    exploration; preserve observed chroma 4:2:0/4:4:4 and progressive flag.
  H.264: sampled effective_crf_interp with +/-0.6 CRF exploration, measured
    GOP length, B-frame presence, profile, pixel format, and FPS.
  HEVC: libx265 average bitrate target based on measured bitrate_ratio times
    current input bitrate. libx264-fitted 'effective_crf_interp' MUST NOT be
    used as a literal x265 CRF. Optional +/-8% bitrate target exploration.
  Brightness/contrast: take sample's measured b and a, add up to +/-10%
    relative exploration, and apply in luma (images) / ffmpeg eq (videos).
    This is a PRE-ENCODING approximation, not a guaranteed calibrated output
    b/a; encoding itself can change these values. Re-measure if essential.
  Instagram crop: center square, with random center jitter 10% of the time,
    up to +/-3% of extra crop margin. Actual crop anchor is not proven by
    alignment shift_x/shift_y (those values are alignment residuals).

OUTPUT:
  output/.../<originalname>__aug000_<platform>_<quality>.<jpg/mp4/orig_ext>
  output/augmentation_manifest.jsonl (one JSON per generated item).

SAFETY/DATASET HYGIENE:
  - pass TRAIN SPLIT input only; never process held-out test set for training.
  - preserve input relative directory layout, labels, and optional image masks.
  - using --copies creates multiple transformed versions of the same example;
    keep grouped by source when splitting to avoid data leakage.
  - masks undergo only geometric operations (nearest-neighbor); no JPEG
    conversion or photometric adjustments are applied to a segmentation mask.
  - video segmentation masks require the SAME spatial and temporal recipe;
    this script writes the recipe to manifest but does not transform masks.
  - all probabilities are explicit experimental choices; 480/480 were measured
    but that does not establish population-level platform prevalence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}
PLATFORMS = ('clean', 'facebook', 'instagram', 'kakao')
IG_DUPLICATE_CHECK_FILES = {
    ('image', 'mid'): 'instagram_mid_image_measurement.csv',
    ('image', 'low'): 'instagram_low_image_measurement.csv',
    ('video', 'mid'): 'instagram_mid_video_measurement.csv',
    ('video', 'low'): 'instagram_low_video_measurement.csv',
}

QUALITY_FILES = {
    ('facebook', 'image', 'high'): 'facebook_high_image_measurement.csv',
    ('facebook', 'image', 'mid'): 'facebook_mid_image_measurement.csv',
    ('facebook', 'image', 'low'): 'facebook_low_image_measurement.csv',
    ('facebook', 'video', 'high'): 'facebook_high_video_measurement.csv',
    ('facebook', 'video', 'mid'): 'facebook_mid_video_measurement.csv',
    ('facebook', 'video', 'low'): 'facebook_low_video_measurement.csv',
    # IMPORTANT: only one Instagram profile for each modality to avoid
    # triple-counting high/mid/low repeated measurement outcomes.
    ('instagram', 'image', 'shared'): 'instagram_high_image_measurement.csv',
    ('instagram', 'video', 'shared'): 'instagram_high_video_measurement.csv',
    ('kakao', 'image', 'original'): 'kakao_original_image_measurement.csv',
    ('kakao', 'image', 'normal'): 'kakao_normal_image_measurement.csv',
    ('kakao', 'image', 'low'): 'kakao_low_image_measurement.csv',
    ('kakao', 'video', 'single'): 'kakao_video_measurement.csv',
}


def fnum(s: Any, default: Optional[float] = None) -> Optional[float]:
    if s is None or str(s).strip().lower() in ('', 'nan', 'none', 'n/a'):
        return default
    try:
        ans = float(s)
        return ans if math.isfinite(ans) else default
    except (ValueError, TypeError):
        return default


def inum(s: Any, default: int = 0) -> int:
    return int(round(fnum(s, default) or 0))


def flag(s: Any, default: bool = False) -> bool:
    return default if s is None or str(s) == '' else str(s).lower() in ('true', '1', 'yes')


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def nearest_even(n: float) -> int:
    return max(2, 2 * int(round(float(n) / 2)))


def stable_seed(seed: int, rel: Path, copy_index: int, epoch: int) -> int:
    digest = hashlib.sha256(
        f'{seed}|{rel.as_posix()}|{copy_index}|{epoch}'.encode('utf8')
    ).digest()
    return int.from_bytes(digest[:8], 'big')


def band_from_dimensions(modality: str, w: int, h: int) -> str:
    # Image filenames in study: ...-le512/le1080/le2048.
    # Video filenames: ...-720p/1080p/2160p.
    if modality == 'image':
        size = max(w, h)
        return 'le512' if size <= 768 else ('le1080' if size <= 1536 else 'le2048')
    # Videos include landscape 1920x1080 and portrait 1080x1920,
    # so classification must use the SHORT edge, not the long edge.
    short = min(w, h)
    return '720p' if short <= 900 else ('1080p' if short <= 1536 else '2160p')


def row_band(r: dict[str, str]) -> str:
    return r['stimulus_id'].rsplit('-', 1)[-1]


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)


def ffprobe_json(path: Path) -> dict[str, Any]:
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,avg_frame_rate,bit_rate,nb_frames,codec_name,profile',
           '-show_entries', 'format=bit_rate,duration', '-of', 'json', str(path)]
    obj = json.loads(run(cmd).stdout)
    streams = obj.get('streams', [])
    if not streams:
        raise RuntimeError(f'No video stream: {path}')
    st = streams[0]
    rate = st.get('avg_frame_rate', '0/1')
    if '/' in str(rate):
        n, d = str(rate).split('/', 1)
        fps = float(n) / float(d) if float(d) else 0
    else:
        fps = float(rate)
    # A stream bit_rate may be missing: format bitrate can include audio,
    # and is used as a documented fallback approximation.
    in_br = fnum(st.get('bit_rate'), fnum(obj.get('format', {}).get('bit_rate'), 0))
    return {
        'width': int(st['width']), 'height': int(st['height']),
        'fps': fps, 'bitrate_bps': in_br,
        'codec': st.get('codec_name'), 'profile': st.get('profile'),
        'frames': st.get('nb_frames'),
    }


class MeasurementProfiles:
    def __init__(self, root: Path):
        self.root = root
        self.data: dict[tuple[str, str, str], dict[str, list[dict[str, str]]]] = {}
        missing = [n for n in list(QUALITY_FILES.values()) + list(IG_DUPLICATE_CHECK_FILES.values())
                   if not (root / n).is_file()]
        if missing:
            raise FileNotFoundError(
                f'{len(missing)} required CSV(s) missing in {root}: {missing}')
        for key, name in QUALITY_FILES.items():
            with (root / name).open(encoding='utf-8-sig', newline='') as h:
                rows = list(csv.DictReader(h))
            if len(rows) != 30:
                raise ValueError(f'{name}: expected 30 measured pairs, got {len(rows)}')
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for r in rows:
                groups[row_band(r)].append(r)
            for b, g in groups.items():
                if len(g) != 10:
                    raise ValueError(f'{name}:{b}: expected 10 rows, got {len(g)}')
            self.data[key] = dict(groups)
        # Verify every supplied Instagram mid/low CSV really is equivalent
        # in the material transformation fields before merging profiles.
        # The raw download hashes are NOT stored in these measurement CSVs.
        semantic_fields = ('reencode_verdict','orig_width','orig_height',
                           'down_width','down_height','crop_applied',
                           'down_estimated_jpeg_qf','down_chroma_subsampling',
                           'down_jpeg_progressive','down_codec_name',
                           'down_profile','down_avg_frame_rate','down_gop_mean',
                           'effective_crf_interp','bitrate_ratio',
                           'brightness_offset_b','contrast_gain_a',
                           'displayed_brightness_offset_b_median',
                           'displayed_contrast_gain_a_median')
        for (modality, quality), filename in IG_DUPLICATE_CHECK_FILES.items():
            with (root / filename).open(encoding='utf-8-sig', newline='') as h:
                duplicates = list(csv.DictReader(h))
            primary = {r['stimulus_id']: r for group in self.data[('instagram', modality, 'shared')].values()
                       for r in group}
            if len(duplicates) != 30 or len(primary) != 30:
                raise ValueError(f'Instagram duplicate validation failed: {filename}: not 30 rows')
            for r in duplicates:
                ref = primary.get(r['stimulus_id'])
                if ref is None or any(ref.get(k,'') != r.get(k,'') for k in semantic_fields):
                    raise ValueError(
                        f'{filename} differs for {r["stimulus_id"]}; cannot safely collapse qualities.')

    def show(self):
        print('PROFILE  BAND    n   reencode_verdict:count')
        for (plat, modality, quality), groups in self.data.items():
            for b, rows in sorted(groups.items()):
                ver = dict(Counter(r['reencode_verdict'] for r in rows))
                print(f'{plat:9} {modality:5} {quality:8} {b:6} {len(rows):2}  {ver}')

    def choose(self, platform: str, modality: str, quality: str,
               band: str, width: int, height: int, rng: random.Random
               ) -> tuple[dict[str, str], str]:
        groups = self.data[(platform, modality, quality)]
        used_band = band
        if used_band not in groups:
            raise ValueError(f'Band {band!r} not recorded for {platform}/{modality}/{quality}')
        rows = groups[used_band]
        # The observations can be STRONGLY dependent on input aspect.
        # Kakao 720p video is a key example: landscape -> geometry preserved,
        # portrait -> resized. Do not assign the unconditional 7:3 frequency
        # to every orientation. Condition on the nearest measured aspect group.
        aspect = width / height
        scored = []
        for r in rows:
            ref_aspect = inum(r['orig_width']) / max(1, inum(r['orig_height']))
            err = abs(math.log(max(1e-6, aspect / ref_aspect)))
            same_orientation = (width >= height) == (inum(r['orig_width']) >= inum(r['orig_height']))
            if same_orientation:
                scored.append((err, r))
        if not scored:  # out-of-distribution orientation: fall back with warning
            scored = [(abs(math.log(max(1e-6, aspect/(inum(r['orig_width'])/max(1,inum(r['orig_height'])))))), r)
                      for r in rows]
        closest = min(err for err, _ in scored)
        similar = [r for err, r in scored if err <= closest + 0.10]
        return rng.choice(similar), used_band


def pick_quality(platform: str, modality: str, rng: random.Random,
                 forced: Optional[str] = None) -> str:
    if platform == 'instagram':
        if forced not in (None, 'shared', 'high', 'mid', 'low'):
            raise ValueError('Instagram uses a collapsed shared profile.')
        return 'shared'
    if platform == 'kakao' and modality == 'video':
        if forced not in (None, 'single'):
            raise ValueError('KakaoTalk video only has one real measured profile.')
        return 'single'
    if platform == 'facebook' and modality == 'video':
        opts = ['high', 'mid', 'low']
        probs = [0.50, 0.25, 0.25]
    else:
        opts = ['high', 'mid', 'low'] if platform == 'facebook' else ['original', 'normal', 'low']
        probs = [1/3] * 3
    if forced is not None:
        if forced not in opts:
            raise ValueError(f'Invalid quality {forced} for {platform}/{modality}: {opts}')
        return forced
    return rng.choices(opts, probs, k=1)[0]


def photo_targets(row: dict[str, str], modality: str,
                  rng: random.Random, args: argparse.Namespace) -> dict[str, float]:
    if modality == 'image':
        b = fnum(row.get('brightness_offset_b'), 0.0) or 0.0
        a = fnum(row.get('contrast_gain_a'), 1.0) or 1.0
    else:
        b = fnum(row.get('displayed_brightness_offset_b_median'), 0.0) or 0.0
        a = fnum(row.get('displayed_contrast_gain_a_median'), 1.0) or 1.0
    if rng.random() <= args.p_photo:
        # Proposal: slight +/-10% scatter around MEASURED row values.
        jitter = args.photo_jitter
        b *= rng.uniform(1-jitter, 1+jitter)
        a = 1.0 + (a - 1.0) * rng.uniform(1-jitter, 1+jitter)
        return {'b': round(clamp(b, -0.04, 0.04), 7),
                'a': round(clamp(a, 0.95, 1.05), 7), 'applied': True}
    return {'b': 0.0, 'a': 1.0, 'applied': False}


def image_geometry(w: int, h: int, row: dict[str, str],
                   platform: str, rng: random.Random, args: argparse.Namespace
                   ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int], dict]:
    # Reference dimensions represent uploaded->redownloaded mapping.
    ow = inum(row['orig_width']); oh = inum(row['orig_height'])
    dw = inum(row['down_width']); dh = inum(row['down_height'])
    crop = flag(row.get('crop_applied')) and platform == 'instagram'
    info = {'reference_size': [ow, oh], 'reference_download_size': [dw, dh],
            'measured_crop_applied': flag(row.get('crop_applied'))}
    if crop:
        square = min(w, h)
        span_x, span_y = w - square, h - square
        offx, offy = span_x / 2, span_y / 2
        jitter = rng.random() < args.ig_crop_jitter_prob
        if jitter:
            offx += rng.uniform(-1, 1) * span_x * args.ig_crop_jitter_frac
            offy += rng.uniform(-1, 1) * span_y * args.ig_crop_jitter_frac
        x = max(0, min(w - square, int(round(offx))))
        y = max(0, min(h - square, int(round(offy))))
        rect = (x, y, x + square, y + square)
        # For the observed images, output square side is generally the INPUT
        # short side; proportional adjustment accommodates other source sizes.
        short_ref = min(ow, oh)
        side_ratio = dw / max(1, short_ref)
        side = max(1, int(round(square * side_ratio)))
        info.update({'center_crop_assumed': True, 'crop_jittered': jitter,
                     'crop_box': list(rect), 'resize_to': [side, side]})
        return rect, (side, side), info
    # For an identity verdict, preserve geometry. JPEG reencode with no
    # resized output will still take this no-crop path.
    nw = max(1, int(round(w * dw / max(1, ow))))
    nh = max(1, int(round(h * dh / max(1, oh))))
    info.update({'center_crop_assumed': False, 'crop_jittered': False,
                 'crop_box': None, 'resize_to': [nw, nh]})
    return None, (nw, nh), info


def adjust_luma_pillow(img, a: float, b: float):
    """Approximate measured y=a*x+b on normalized 8-bit luma."""
    from PIL import Image
    if abs(a-1.0) < 1e-12 and abs(b) < 1e-12:
        return img
    yc = img.convert('YCbCr')
    y, cb, cr = yc.split()
    lut = [max(0, min(255, int(round(255*(a*i/255+b))))) for i in range(256)]
    return Image.merge('YCbCr', (y.point(lut), cb, cr)).convert('RGB')


def image_mask_for(image: Path, source_root: Path, mask_root: Path) -> Optional[Path]:
    rel = image.relative_to(source_root)
    m = mask_root / rel.parent / (rel.stem + '.png')
    return m if m.is_file() else None


def transform_image(src: Path, dst: Path, row: dict[str,str],
                    platform: str, rng: random.Random, args: argparse.Namespace,
                    mask_src: Optional[Path], mask_dst: Optional[Path]) -> dict:
    from PIL import Image, ImageOps
    verdict = row['reencode_verdict']
    if verdict == 'container_or_metadata_only':
        # Pixel-level identity: do not mistakenly introduce a JPEG generation.
        shutil.copy2(src, dst)
        if mask_src and mask_dst:
            mask_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_src, mask_dst)
        return {'verdict': verdict, 'pixel_transform': 'identity',
                'photo': {'b': 0, 'a': 1, 'applied': False}, 'jpeg': None,
                'geometry': 'preserved',
                'note': 'Metadata/container rewrite not emulated; output byte-copy.'}

    with Image.open(src) as open_img:
        im = ImageOps.exif_transpose(open_img).convert('RGB')
        orig_icc = open_img.info.get('icc_profile')
    crop_rect, target, geo = image_geometry(im.width, im.height, row, platform, rng, args)
    if crop_rect is not None:
        im = im.crop(crop_rect)
    if im.size != target:
        im = im.resize(target, Image.Resampling.LANCZOS)
    photo = photo_targets(row, 'image', rng, args)
    if photo['applied']:
        im = adjust_luma_pillow(im, photo['a'], photo['b'])
    qf = inum(row.get('down_estimated_jpeg_qf'), 95)
    qf = int(round(clamp(qf + rng.uniform(-args.jpeg_qf_jitter, args.jpeg_qf_jitter), 1, 100)))
    subsamp = row.get('down_chroma_subsampling', '4:2:0')
    subsampling = {'4:4:4': 0, '4:2:2': 1, '4:2:0': 2}.get(subsamp, 2)
    progressive = flag(row.get('down_jpeg_progressive'))
    kwargs: dict[str, Any] = {'format': 'JPEG', 'quality': qf,
                               'subsampling': subsampling, 'progressive': progressive}
    if flag(row.get('down_has_icc')) and orig_icc:
        kwargs['icc_profile'] = orig_icc
    im.save(dst, **kwargs)
    if mask_src and mask_dst:
        # Ensure masks are transformed IDENTICALLY in space, but never
        # compressed as JPEG, color-corrected, or interpolated bilinearly.
        with Image.open(mask_src) as mask_in:
            mask = mask_in.copy()
        if crop_rect is not None:
            mask = mask.crop(crop_rect)
        if mask.size != target:
            mask = mask.resize(target, Image.Resampling.NEAREST)
        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_dst)
    return {'verdict': verdict, 'pixel_transform': 'resized_or_recompressed',
            'geometry': geo, 'photo': photo,
            'jpeg': {'qf': qf, 'subsampling': subsamp, 'progressive': progressive,
                     'metadata_preserved': False}}


def transform_video(src: Path, dst: Path, row: dict[str,str], platform: str,
                    rng: random.Random, args: argparse.Namespace,
                    probe: dict[str, Any]) -> dict:
    src_w, src_h = probe['width'], probe['height']
    ow = inum(row['orig_width']); oh = inum(row['orig_height'])
    dw = inum(row['down_width']); dh = inum(row['down_height'])
    filters: list[str] = []
    geometry: dict[str, Any] = {'reference_size': [ow, oh],
                               'reference_download_size': [dw, dh]}
    if platform == 'instagram':
        # IG videos 720x720 in all 30 measured pairs, per quality profile.
        # Center anchoring is an APPROXIMATION, not proven by CSV.
        min_edge = min(src_w, src_h)
        off_x = (src_w - min_edge) // 2
        off_y = (src_h - min_edge) // 2
        jitter = rng.random() < args.ig_crop_jitter_prob
        if jitter:
            off_x = round(off_x + rng.uniform(-1, 1) * (src_w-min_edge) * args.ig_crop_jitter_frac)
            off_y = round(off_y + rng.uniform(-1, 1) * (src_h-min_edge) * args.ig_crop_jitter_frac)
        off_x = max(0, min(src_w-min_edge, int(off_x)))
        off_y = max(0, min(src_h-min_edge, int(off_y)))
        filters.append(f'crop={min_edge}:{min_edge}:{off_x}:{off_y}')
        target = (nearest_even(dw), nearest_even(dh))  # measured 720x720
        geometry.update({'crop_box': [off_x, off_y, min_edge, min_edge],
                         'crop_jittered': jitter, 'center_crop_assumed': True})
    else:
        target = (nearest_even(src_w*dw/max(1,ow)),
                  nearest_even(src_h*dh/max(1,oh)))
        geometry.update({'crop_box': None, 'center_crop_assumed': False})
    if target != (src_w, src_h) or platform == 'instagram':
        filters.append(f'scale={target[0]}:{target[1]}:flags=lanczos')
    geometry['output_size'] = list(target)

    photo = photo_targets(row, 'video', rng, args)
    if photo['applied'] and (abs(photo['a']-1)>1e-7 or abs(photo['b'])>1e-7):
        # FFmpeg eq brightness/contrast ranges are analogous but not identical
        # to the regression coefficients on decoded/displayed Y. Re-measure.
        filters.append(f"eq=contrast={photo['a']:.7f}:brightness={photo['b']:.7f}")

    in_fps = probe['fps']
    out_fps = fnum(row.get('down_avg_frame_rate'), in_fps) or in_fps
    # Match observed 1-frame leading offset for Kakao 1080p and 2160p.
    drop_frames = max(0, inum(row.get('temporal_offset_frames'), 0))
    if drop_frames and abs(in_fps-out_fps)<0.05:
        filters.append(f'trim=start_frame={drop_frames}')
        filters.append('setpts=PTS-STARTPTS')
    if in_fps > 0 and abs(in_fps-out_fps) >= 0.05:
        filters.append(f'fps=fps={out_fps:.8g}')
    codec = str(row.get('down_codec_name', 'h264')).lower()
    is_hevc = codec in ('hevc', 'h265')
    codec_args = ['-c:v', 'libx265' if is_hevc else 'libx264',
                  '-preset', args.video_preset, '-pix_fmt', 'yuv420p']
    observed_gop = inum(row.get('down_gop_mean'), 150)
    gop = max(1, observed_gop)
    bframes = 2 if flag(row.get('down_has_b_frames')) else 0
    codec_args.extend(['-g', str(gop), '-keyint_min', str(gop),
                       '-sc_threshold', '0', '-bf', str(bframes)])
    profile = str(row.get('down_profile', '')).lower()
    bitrate_target_bps = None
    if is_hevc:
        ratio = fnum(row.get('bitrate_ratio'), 0.3) or 0.3
        ratio *= rng.uniform(1-args.hevc_bitrate_jitter, 1+args.hevc_bitrate_jitter)
        input_bitrate = probe['bitrate_bps'] or None
        if not input_bitrate:
            raise ValueError('HEVC bitrate matching requires input video bitrate from ffprobe')
        bitrate_target_bps = max(100_000, int(input_bitrate * ratio))
        # x264-fitted effective_crf_interp in CSV is NOT a legitimate x265 CRF.
        codec_args.extend(['-b:v', str(bitrate_target_bps),
                           '-tag:v', 'hvc1'])
        if profile in ('main', 'main10'):
            codec_args.extend(['-profile:v', 'main'])
    else:
        ref_crf = fnum(row.get('effective_crf_interp'), 26.0) or 26.0
        crf = clamp(ref_crf + rng.uniform(-args.h264_crf_jitter,
                                           args.h264_crf_jitter), 0, 51)
        codec_args.extend(['-crf', f'{crf:.2f}'])
        if profile in ('high','main','baseline'):
            codec_args.extend(['-profile:v', profile])

    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
           '-y', '-i', str(src), '-map', '0:v:0', '-map', '0:a?']
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    cmd += codec_args
    # Explicit output rate stabilizes timing after a one-frame trim.
    cmd += ['-r', f'{out_fps:.8g}', '-fps_mode', 'cfr', '-color_range', 'tv', '-c:a', 'aac',
            '-b:a', '128k', '-movflags', '+faststart', str(dst)]
    try:
        run(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'ffmpeg failed: {exc.stderr[-1600:]}') from exc
    out_probe = ffprobe_json(dst) if not args.skip_video_output_probe else None
    return {'verdict': row['reencode_verdict'], 'geometry': geometry,
            'photo': photo, 'video': {
                'codec': codec, 'profile': profile, 'fps': out_fps,
                'input_fps': in_fps, 'reference_gop': gop,
                'bframes': bframes,
                'h264_crf': (None if is_hevc else round(crf, 3)),
                'hevc_target_bitrate_bps': bitrate_target_bps,
                'reference_bitrate_ratio': fnum(row.get('bitrate_ratio')),
                'input_bitrate_bps': probe['bitrate_bps'],
                'output_probe': out_probe,
                'bitrate_match_warning': (
                    'Target bitrate ratio not reached; recalibrate on full clips.'
                    if is_hevc and out_probe and probe['bitrate_bps'] and
                    out_probe['bitrate_bps'] and bitrate_target_bps and
                    abs(out_probe['bitrate_bps'] - bitrate_target_bps)/bitrate_target_bps > 0.30
                    else None),
                'actual_bitrate_ratio': (
                    round(out_probe['bitrate_bps']/probe['bitrate_bps'], 5)
                    if out_probe and probe['bitrate_bps'] and out_probe['bitrate_bps']
                    else None),
                'temporal_offset_frames_used': drop_frames,
                'ffmpeg_filter': ','.join(filters),
            }}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--measurements-dir', type=Path, required=True,
                    help='Folder holding the original 16 *_measurement.csv files')
    ap.add_argument('--input', type=Path, help='Train media root (recursive scan)')
    ap.add_argument('--output', type=Path, help='Augmented media root')
    ap.add_argument('--mask-root', type=Path, help='Optional image mask root, same relative paths and .png stem')
    ap.add_argument('--mask-output', type=Path, help='Required when --mask-root is supplied')
    ap.add_argument('--copies', type=int, default=1, help='Independent sampled augmentations per input (default 1)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epoch', type=int, default=0)
    ap.add_argument('--p-clean', type=float, default=0.25)
    ap.add_argument('--p-facebook', type=float, default=0.25)
    ap.add_argument('--p-instagram', type=float, default=0.25)
    ap.add_argument('--p-kakao', type=float, default=0.25)
    ap.add_argument('--p-photo', type=float, default=1.0,
                    help='Apply measured small brightness/contrast target on reencoded items')
    ap.add_argument('--photo-jitter', type=float, default=0.10,
                    help='Relative variation around measured b and (a-1). Proposed, not measured.')
    ap.add_argument('--jpeg-qf-jitter', type=float, default=1.0)
    ap.add_argument('--h264-crf-jitter', type=float, default=0.6)
    ap.add_argument('--hevc-bitrate-jitter', type=float, default=0.08)
    ap.add_argument('--ig-crop-jitter-prob', type=float, default=0.10)
    ap.add_argument('--ig-crop-jitter-frac', type=float, default=0.03,
                    help='Fraction of EXTRA crop margin for displacement')
    ap.add_argument('--video-preset', default='medium', choices=['ultrafast','superfast','veryfast',
                    'faster','fast','medium','slow','slower','veryslow'])
    ap.add_argument('--skip-video-output-probe', action='store_true')
    ap.add_argument('--force-platform', choices=PLATFORMS,
                    help='Debug / ablation: override all platform probabilities')
    ap.add_argument('--force-quality', help='Debug / ablation: force platform quality if valid')
    ap.add_argument('--show-profiles', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='Plan and write manifest but do not transcode or copy')
    ap.add_argument('--continue-on-error', action='store_true')
    return ap.parse_args()


def validate(a: argparse.Namespace):
    if a.show_profiles:
        return
    if not a.input or not a.output:
        raise ValueError('--input and --output are required unless --show-profiles')
    if a.copies <= 0:
        raise ValueError('--copies must be >=1')
    if a.mask_root and not a.mask_output:
        raise ValueError('--mask-root requires --mask-output')
    if (a.mask_root and a.mask_output) and a.mask_root.resolve() == a.mask_output.resolve():
        raise ValueError('mask input and output roots must differ')
    probs = [a.p_clean, a.p_facebook, a.p_instagram, a.p_kakao]
    if min(probs) < 0 or abs(sum(probs) - 1) > 1e-8:
        raise ValueError('Platform probabilities must be >=0 and sum to 1.0')
    if not (0 <= a.p_photo <= 1 and 0 <= a.ig_crop_jitter_prob <= 1):
        raise ValueError('Probabilities must be in [0,1]')
    if any(x < 0 or x > 1 for x in [a.photo_jitter, a.jpeg_qf_jitter/100,
                                   a.hevc_bitrate_jitter, a.ig_crop_jitter_frac]):
        raise ValueError('Invalid jitter; check parameter ranges')
    if a.h264_crf_jitter < 0 or a.h264_crf_jitter > 10:
        raise ValueError('h264_crf_jitter must be between 0 and 10')
    if not a.input.is_dir():
        raise ValueError(f'Input is not a directory: {a.input}')
    if a.output.resolve() == a.input.resolve():
        raise ValueError('Output must be separate from input')
    if a.input.resolve() in a.output.resolve().parents:
        raise ValueError('Do not place output inside input (recursive scan contamination)')


def main():
    args = parse_args()
    policy = MeasurementProfiles(args.measurements_dir)
    if args.show_profiles:
        policy.show()
        return
    validate(args)
    args.output.mkdir(parents=True, exist_ok=True)
    weights = [args.p_clean, args.p_facebook, args.p_instagram, args.p_kakao]
    print('Platform probability:', dict(zip(PLATFORMS, weights)))
    inputs = [p for p in sorted(args.input.rglob('*')) if p.is_file()
              and p.suffix.lower() in (IMAGE_SUFFIXES|VIDEO_SUFFIXES)]
    if not inputs:
        raise RuntimeError(f'No image/video files found in {args.input}')
    manifest = args.output / 'augmentation_manifest.jsonl'
    counts: Counter[str] = Counter()
    with manifest.open('w', encoding='utf8') as writer:
        for i, src in enumerate(inputs, 1):
            rel = src.relative_to(args.input)
            modality = 'image' if src.suffix.lower() in IMAGE_SUFFIXES else 'video'
            try:
                if modality == 'image':
                    from PIL import Image, ImageOps
                    with Image.open(src) as im:
                        # PIL dimensions without orientation changes. Upload
                        # reference sizes are measured after decode; most sources
                        # already have exif orientation=1.
                        trans = ImageOps.exif_transpose(im)
                        w, h = trans.size
                    probe = None
                else:
                    probe = ffprobe_json(src)
                    w, h = probe['width'], probe['height']
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print('WARN read error', src, exc, file=sys.stderr)
                continue
            band = band_from_dimensions(modality, w, h)
            for copy_index in range(args.copies):
                rng = random.Random(stable_seed(args.seed, rel, copy_index, args.epoch))
                platform = args.force_platform or rng.choices(PLATFORMS, weights, k=1)[0]
                record: dict[str, Any] = {
                    'input': str(src), 'source_relative_path': rel.as_posix(),
                    'copy_index': copy_index, 'seed': args.seed, 'epoch': args.epoch,
                    'modality': modality, 'input_size': [w, h],
                    'input_band': band, 'platform': platform,
                    'conditional_quality_probabilities': (
                        {'high':0.5,'mid':0.25,'low':0.25}
                        if platform=='facebook' and modality=='video' else
                        {'high':1/3,'mid':1/3,'low':1/3}
                        if platform=='facebook' else
                        {'original':1/3,'normal':1/3,'low':1/3}
                        if platform=='kakao' and modality=='image' else {'shared':1.0}
                        if platform=='instagram' else {'single':1.0} if platform=='kakao' else {}),
                }
                try:
                    if platform == 'clean':
                        if args.force_quality:
                            raise ValueError('Cannot force quality on clean route')
                        quality, row = 'none', None
                        out_suffix = src.suffix
                    else:
                        quality = pick_quality(platform, modality, rng, args.force_quality)
                        row, _ = policy.choose(platform, modality, quality, band, w, h, rng)
                        record.update({'quality': quality,
                                       'measured_reference_stimulus_id': row['stimulus_id'],
                                       'measured_reference_target_id': row.get('target_id'),
                                       'measured_reference_verdict': row['reencode_verdict'],
                                       'reference_measurements_csv': QUALITY_FILES[(platform,modality,quality)]})
                        verdict = row['reencode_verdict']
                        out_suffix = (src.suffix if modality == 'image' and
                                      verdict == 'container_or_metadata_only' else
                                      '.jpg' if modality == 'image' else '.mp4')
                    dst = args.output / rel.parent / (f'{rel.stem}__aug{copy_index:03d}_'
                                         f'{platform}_{quality}{out_suffix}')
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    record['output'] = str(dst)
                    mask_src = (image_mask_for(src, args.input, args.mask_root)
                                if modality == 'image' and args.mask_root else None)
                    mask_dst = (args.mask_output / rel.parent /
                                f'{rel.stem}__aug{copy_index:03d}_{platform}_{quality}.png'
                                if mask_src and args.mask_output else None)
                    if mask_src:
                        record['input_mask'] = str(mask_src)
                        record['output_mask'] = str(mask_dst)
                    if args.dry_run:
                        record['status'] = 'dry_run'
                    elif platform == 'clean':
                        shutil.copy2(src, dst)
                        if mask_src and mask_dst:
                            mask_dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(mask_src, mask_dst)
                        record['transformation'] = 'bytewise_copy'
                        record['status'] = 'ok'
                    elif modality == 'image':
                        record['transformation'] = transform_image(
                            src, dst, row, platform, rng, args, mask_src, mask_dst)
                        record['status'] = 'ok'
                    else:
                        record['transformation'] = transform_video(
                            src, dst, row, platform, rng, args, probe)
                        record['status'] = 'ok'
                    counts[f'{platform}/{quality}/{band}'] += 1
                except Exception as exc:
                    record['status'] = 'error'
                    record['error'] = str(exc)
                    if not args.continue_on_error:
                        writer.write(json.dumps(record, ensure_ascii=False)+'\n')
                        writer.flush()
                        raise
                    print('WARN transform error', src, str(exc), file=sys.stderr)
                writer.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(f'Processed {len(inputs)} source files, {args.copies} copy/copies each')
    print('Output manifest:', manifest)
    print('Route counts:')
    for key, n in sorted(counts.items()):
        print(f'  {key:34} {n}')
    print('IMPORTANT: Measured reference values do not guarantee identical'
          ' downstream JPEG/codec artifacts. Re-measure sample outputs.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print('ERROR:', e, file=sys.stderr)
        sys.exit(2)