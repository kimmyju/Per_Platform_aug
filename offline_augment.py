#!/usr/bin/env python3
"""Offline platform-aware augmentation.

This script uses the shared `augmentation_profiles.py` catalog, which was created
once from measured CSVs. Runtime does NOT read measurement.csv. For each input file,
it chooses one pre-built recipe and writes an augmented output file to disk.

Images: Crop -> Resize -> Brightness/Contrast -> JPEG re-encode
Videos: Crop/Scale -> FPS/Trim -> Brightness/Contrast -> H.264/HEVC transcode
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

from augmentation_profiles import (
    IMAGE_SUFFIXES, VIDEO_SUFFIXES,
    PLATFORM_MIXTURE_DEFAULT, QUALITY_WEIGHTS_DEFAULT,
    sample_policy, inum, fnum, flag, clamp, nearest_even,
)


def run(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def ffprobe_json(path: Path):
    cmd = [
        'ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', str(path)
    ]
    proc = run(cmd)
    data = json.loads(proc.stdout)
    streams = data.get('streams', [])
    v = next((s for s in streams if s.get('codec_type') == 'video'), None)
    if v is None:
        raise ValueError(f'No video stream found in {path}')
    width, height = int(v['width']), int(v['height'])
    afr = v.get('avg_frame_rate', '0/1')
    num, den = afr.split('/') if '/' in afr else (afr, '1')
    fps = (float(num) / float(den)) if float(den) != 0 else 0.0
    bitrate = int(v.get('bit_rate') or data.get('format', {}).get('bit_rate') or 0)
    return {'width': width, 'height': height, 'fps': fps, 'bitrate_bps': bitrate, 'streams': streams, 'format': data.get('format', {})}


def adjust_luma_pillow(image, a: float, b: float):
    """Apply measured photometric affine shift to Y, not independently to RGB.

    This is an approximate *one-pass* application of measured a,b, NOT a
    calibrated copy of the platform color pipeline.
    """
    from PIL import Image
    if abs(a - 1.0) < 1e-8 and abs(b) < 1e-8:
        return image
    ycbcr = image.convert('YCbCr')
    y, cb, cr = ycbcr.split()
    lut = [int(round(clamp(a*v + b*255.0, 0, 255))) for v in range(256)]
    return Image.merge('YCbCr', (y.point(lut), cb, cr)).convert('RGB')


def choose_photo(recipe: dict, modality: str, p_photo: float, photo_jitter: float, rng):
    if modality == 'image':
        b = fnum(recipe.get('brightness_offset_b'), 0.0) or 0.0
        a = fnum(recipe.get('contrast_gain_a'), 1.0) or 1.0
    else:
        b = fnum(recipe.get('brightness_offset_b'), 0.0) or 0.0
        a = fnum(recipe.get('contrast_gain_a'), 1.0) or 1.0
    if rng.random() <= p_photo:
        b *= rng.uniform(1-photo_jitter, 1+photo_jitter)
        a = 1.0 + (a - 1.0) * rng.uniform(1-photo_jitter, 1+photo_jitter)
        return {'applied': True, 'b': round(clamp(b, -0.04, 0.04), 7), 'a': round(clamp(a, 0.95, 1.05), 7)}
    return {'applied': False, 'b': 0.0, 'a': 1.0}


def image_geometry(width: int, height: int, recipe: dict, platform: str, rng, ig_crop_jitter_prob: float, ig_crop_jitter_frac: float):
    ow = inum(recipe.get('orig_width'), width) or width
    oh = inum(recipe.get('orig_height'), height) or height
    dw = inum(recipe.get('down_width'), width) or width
    dh = inum(recipe.get('down_height'), height) or height
    crop_box = None
    if platform == 'instagram':
        side = min(width, height)
        ox = (width - side) // 2
        oy = (height - side) // 2
        jittered = False
        if rng.random() < ig_crop_jitter_prob:
            ox = round(ox + rng.uniform(-1, 1) * (width - side) * ig_crop_jitter_frac)
            oy = round(oy + rng.uniform(-1, 1) * (height - side) * ig_crop_jitter_frac)
            ox = max(0, min(width - side, int(ox)))
            oy = max(0, min(height - side, int(oy)))
            jittered = True
        crop_box = (ox, oy, ox + side, oy + side)
        target = (max(1, dw), max(1, dh))
        geo = {'crop_box': list(crop_box), 'target_size': list(target), 'center_crop_assumed': True, 'crop_jittered': jittered}
    else:
        sx = dw / max(1, ow)
        sy = dh / max(1, oh)
        target = (max(1, round(width * sx)), max(1, round(height * sy)))
        geo = {'crop_box': None, 'target_size': list(target), 'center_crop_assumed': False, 'scale_x': sx, 'scale_y': sy}
    return crop_box, target, geo


def transform_image(src: Path, dst: Path, recipe: dict, platform: str, args, rng):
    from PIL import Image
    with Image.open(src) as im:
        im = im.convert('RGB')
        if recipe.get('reencode_verdict') == 'container_or_metadata_only':
            shutil.copy2(src, dst)
            return {'execution': 'identity-copy', 'note': 'pixel-preserving approximation for metadata/container-only case'}
        crop_box, target, geo = image_geometry(im.width, im.height, recipe, platform, rng, args.ig_crop_jitter_prob, args.ig_crop_jitter_frac)
        if crop_box is not None:
            im = im.crop(crop_box)
        if im.size != target:
            im = im.resize(target, Image.Resampling.LANCZOS)
        photo = choose_photo(recipe, 'image', args.p_photo, args.photo_jitter, rng)
        if photo['applied']:
            im = adjust_luma_pillow(im, photo['a'], photo['b'])
        qf = inum(recipe.get('down_estimated_jpeg_qf'), 95) or 95
        qf = int(round(clamp(qf + rng.uniform(-args.jpeg_qf_jitter, args.jpeg_qf_jitter), 1, 100)))
        subsamp = recipe.get('down_chroma_subsampling', '4:2:0')
        subsampling = {'4:4:4': 0, '4:2:2': 1, '4:2:0': 2}.get(subsamp, 2)
        progressive = flag(recipe.get('down_jpeg_progressive'))
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=qf, subsampling=subsampling, progressive=progressive)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(buf.getvalue())
        return {'execution': 'jpeg-reencode', 'geometry': geo, 'photo': photo, 'jpeg': {'qf': qf, 'subsampling': subsamp, 'progressive': progressive}}


def transform_video(src: Path, dst: Path, recipe: dict, platform: str, args, rng):
    if recipe.get('reencode_verdict') == 'container_or_metadata_only':
        shutil.copy2(src, dst)
        return {'execution': 'identity-copy', 'note': 'pixel-preserving approximation for metadata/container-only case'}
    probe = ffprobe_json(src)
    src_w, src_h = probe['width'], probe['height']
    ow = inum(recipe.get('orig_width'), src_w) or src_w
    oh = inum(recipe.get('orig_height'), src_h) or src_h
    dw = inum(recipe.get('down_width'), src_w) or src_w
    dh = inum(recipe.get('down_height'), src_h) or src_h
    filters = []
    if platform == 'instagram':
        side = min(src_w, src_h)
        ox = (src_w - side) // 2
        oy = (src_h - side) // 2
        filters.append(f'crop={side}:{side}:{ox}:{oy}')
        target = (nearest_even(dw), nearest_even(dh))
        geometry = {'crop_box': [ox, oy, side, side], 'output_size': list(target), 'center_crop_assumed': True}
    else:
        target = (nearest_even(src_w * dw / max(1, ow)), nearest_even(src_h * dh / max(1, oh)))
        geometry = {'crop_box': None, 'output_size': list(target), 'center_crop_assumed': False}
    if target != (src_w, src_h) or platform == 'instagram':
        filters.append(f'scale={target[0]}:{target[1]}:flags=lanczos')

    photo = choose_photo(recipe, 'video', args.p_photo, args.photo_jitter, rng)
    if photo['applied']:
        filters.append(f'lutyuv=y=val*{photo["a"]:.7f}+{photo["b"]*219:.7f}')

    in_fps = probe['fps']
    out_fps = fnum(recipe.get('down_avg_frame_rate'), in_fps) or in_fps
    drop_frames = max(0, inum(recipe.get('temporal_offset_frames'), 0) or 0)
    if drop_frames and abs(in_fps - out_fps) < 0.05:
        filters.append(f'trim=start_frame={drop_frames}')
        filters.append('setpts=PTS-STARTPTS')
    if in_fps > 0 and abs(in_fps - out_fps) >= 0.05:
        filters.append(f'fps=fps={out_fps:.8g}')

    codec = str(recipe.get('down_codec_name', 'h264')).lower()
    is_hevc = codec in ('hevc', 'h265')
    codec_args = ['-c:v', 'libx265' if is_hevc else 'libx264', '-preset', args.video_preset, '-pix_fmt', 'yuv420p']
    gop = max(1, inum(recipe.get('down_gop_mean'), 150) or 150)
    bframes = 2 if flag(recipe.get('down_has_b_frames')) else 0
    codec_args.extend(['-g', str(gop), '-keyint_min', str(gop), '-sc_threshold', '0', '-bf', str(bframes)])
    profile = str(recipe.get('down_profile', '')).lower()
    bitrate_target_bps = None
    if is_hevc:
        ratio = fnum(recipe.get('bitrate_ratio'), 0.3) or 0.3
        ratio *= rng.uniform(1 - args.hevc_bitrate_jitter, 1 + args.hevc_bitrate_jitter)
        input_bitrate = probe['bitrate_bps']
        if not input_bitrate:
            raise ValueError('HEVC offline augmentation requires input bitrate from ffprobe')
        bitrate_target_bps = max(100_000, int(input_bitrate * ratio))
        codec_args.extend(['-b:v', str(bitrate_target_bps), '-tag:v', 'hvc1'])
        if profile in ('main', 'main10'):
            codec_args.extend(['-profile:v', 'main'])
    else:
        ref_crf = fnum(recipe.get('effective_crf_interp'), 26.0) or 26.0
        crf = clamp(ref_crf + rng.uniform(-args.h264_crf_jitter, args.h264_crf_jitter), 0, 51)
        codec_args.extend(['-crf', f'{crf:.2f}'])
        if profile in ('high', 'main', 'baseline'):
            codec_args.extend(['-profile:v', profile])

    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(src), '-map', '0:v:0', '-map', '0:a?']
    if filters:
        cmd += ['-vf', ','.join(filters)]
    cmd += codec_args + ['-r', f'{out_fps:.8g}', '-fps_mode', 'cfr', '-color_range', 'tv', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', str(dst)]
    run(cmd)
    return {'execution': 'video-transcode', 'geometry': geometry, 'photo': photo, 'video': {'codec': codec, 'profile': profile, 'out_fps': out_fps, 'gop': gop, 'bframes': bframes, 'bitrate_target_bps': bitrate_target_bps, 'effective_crf_reference': fnum(recipe.get('effective_crf_interp'), None)}}


def gather_files(input_dir: Path):
    items = []
    for p in sorted(input_dir.rglob('*')):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in IMAGE_SUFFIXES or suf in VIDEO_SUFFIXES:
            items.append(p)
    return items


def parse_args():
    ap = argparse.ArgumentParser(description='Offline platform-aware augmentation using pre-built profiles')
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--copies', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epoch', type=int, default=0)
    ap.add_argument('--p-clean', type=float, default=0.25)
    ap.add_argument('--p-facebook', type=float, default=0.25)
    ap.add_argument('--p-instagram', type=float, default=0.25)
    ap.add_argument('--p-kakao', type=float, default=0.25)
    ap.add_argument('--p-photo', type=float, default=1.0)
    ap.add_argument('--photo-jitter', type=float, default=0.0)
    ap.add_argument('--jpeg-qf-jitter', type=float, default=0.0)
    ap.add_argument('--h264-crf-jitter', type=float, default=0.0)
    ap.add_argument('--hevc-bitrate-jitter', type=float, default=0.0)
    ap.add_argument('--ig-crop-jitter-prob', type=float, default=0.0)
    ap.add_argument('--ig-crop-jitter-frac', type=float, default=0.03)
    ap.add_argument('--video-preset', type=str, default='medium')
    ap.add_argument('--force-platform', type=str)
    ap.add_argument('--force-quality', type=str)
    return ap.parse_args()


def main():
    args = parse_args()
    mixture = {'clean': args.p_clean, 'facebook': args.p_facebook, 'instagram': args.p_instagram, 'kakao': args.p_kakao}
    files = gather_files(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'augmentation_manifest.jsonl'
    count = 0
    with manifest.open('w', encoding='utf8') as mf:
        for src in files:
            rel = src.relative_to(args.input)
            for copy_index in range(args.copies):
                suffix = src.suffix.lower()
                key = rel.as_posix()
                if suffix in IMAGE_SUFFIXES:
                    from PIL import Image
                    with Image.open(src) as im:
                        sample = sample_policy('image', im.width, im.height, key=key, base_seed=args.seed, epoch=args.epoch, copy_index=copy_index, force_platform=args.force_platform, force_quality=args.force_quality, mixture=mixture)
                    platform = sample['platform']
                    recipe = sample['recipe']
                    rng = sample['rng']
                    out_ext = rel.suffix if platform == 'clean' or recipe.get('reencode_verdict') == 'container_or_metadata_only' else '.jpg'
                    out_rel = rel.with_name(f'{rel.stem}__aug{copy_index:02d}{out_ext}')
                    dst = args.output / out_rel
                    if platform == 'clean':
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                        exec_info = {'execution': 'identity-copy'}
                    else:
                        exec_info = transform_image(src, dst, recipe, platform, args, rng)
                else:
                    probe = ffprobe_json(src)
                    sample = sample_policy('video', probe['width'], probe['height'], key=key, base_seed=args.seed, epoch=args.epoch, copy_index=copy_index, force_platform=args.force_platform, force_quality=args.force_quality, mixture=mixture)
                    platform = sample['platform']
                    recipe = sample['recipe']
                    rng = sample['rng']
                    out_rel = rel.with_name(f'{rel.stem}__aug{copy_index:02d}{rel.suffix}')
                    dst = args.output / out_rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if platform == 'clean':
                        shutil.copy2(src, dst)
                        exec_info = {'execution': 'identity-copy'}
                    else:
                        exec_info = transform_video(src, dst, recipe, platform, args, rng)
                record = {
                    'source': str(src), 'output': str(dst), 'copy_index': copy_index,
                    'platform': platform, 'quality': sample['quality'], 'band': sample['band'], 'orientation': sample['orientation'],
                    'recipe_id': None if recipe is None else recipe['id'], 'recipe_source_row_ordinal': None if recipe is None else recipe.get('source_row_ordinal'),
                    'recipe_reencode_verdict': None if recipe is None else recipe.get('reencode_verdict'),
                    'execution_info': exec_info,
                }
                mf.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
    print(f'Wrote {count} augmented files to {args.output}')
    print(f'Manifest: {manifest}')


if __name__ == '__main__':
    main()