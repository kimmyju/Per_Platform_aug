#!/usr/bin/env python3
"""Online / runtime platform-aware augmentation.

This module uses pre-built recipes from augmentation_profiles.py. Runtime does not
re-read measurement.csv. It can be used in Dataset.__getitem__() for images and as a
file-based on-the-fly transcode wrapper for videos.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from augmentation_profiles import (
    sample_policy, IMAGE_SUFFIXES, VIDEO_SUFFIXES,
    inum, fnum, flag, clamp, nearest_even,
)


def run(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def ffprobe_json(path: Path):
    proc = run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', str(path)])
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
    return {'width': width, 'height': height, 'fps': fps, 'bitrate_bps': bitrate}


def adjust_luma_pillow(image, a: float, b: float):
    """Apply a one-pass affine Y change; no output measuring/recalibration."""
    from PIL import Image
    if abs(a - 1.0) < 1e-8 and abs(b) < 1e-8:
        return image
    y, cb, cr = image.convert('YCbCr').split()
    lut = [int(round(clamp(a*v + b*255.0, 0, 255))) for v in range(256)]
    return Image.merge('YCbCr', (y.point(lut), cb, cr)).convert('RGB')


@dataclass
class RuntimeConfig:
    seed: int = 42
    epoch: int = 0
    p_clean: float = 0.25
    p_facebook: float = 0.25
    p_instagram: float = 0.25
    p_kakao: float = 0.25
    p_photo: float = 1.0
    photo_jitter: float = 0.0
    jpeg_qf_jitter: float = 0.0
    h264_crf_jitter: float = 0.0
    hevc_bitrate_jitter: float = 0.0
    ig_crop_jitter_prob: float = 0.0
    ig_crop_jitter_frac: float = 0.03
    video_preset: str = 'medium'

    def mixture(self):
        return {'clean': self.p_clean, 'facebook': self.p_facebook, 'instagram': self.p_instagram, 'kakao': self.p_kakao}


def choose_photo(recipe: dict, p_photo: float, photo_jitter: float, rng):
    b = fnum(recipe.get('brightness_offset_b'), 0.0) or 0.0
    a = fnum(recipe.get('contrast_gain_a'), 1.0) or 1.0
    if rng.random() <= p_photo:
        b *= rng.uniform(1-photo_jitter, 1+photo_jitter)
        a = 1.0 + (a - 1.0) * rng.uniform(1-photo_jitter, 1+photo_jitter)
        return {'applied': True, 'b': round(clamp(b, -0.04, 0.04), 7), 'a': round(clamp(a, 0.95, 1.05), 7)}
    return {'applied': False, 'b': 0.0, 'a': 1.0}


def image_geometry(width: int, height: int, recipe: dict, platform: str, rng, cfg: RuntimeConfig):
    ow = inum(recipe.get('orig_width'), width) or width
    oh = inum(recipe.get('orig_height'), height) or height
    dw = inum(recipe.get('down_width'), width) or width
    dh = inum(recipe.get('down_height'), height) or height
    if platform == 'instagram':
        side = min(width, height)
        ox = (width - side) // 2
        oy = (height - side) // 2
        if rng.random() < cfg.ig_crop_jitter_prob:
            ox = round(ox + rng.uniform(-1, 1) * (width - side) * cfg.ig_crop_jitter_frac)
            oy = round(oy + rng.uniform(-1, 1) * (height - side) * cfg.ig_crop_jitter_frac)
            ox = max(0, min(width - side, int(ox)))
            oy = max(0, min(height - side, int(oy)))
            jittered = True
        else:
            jittered = False
        return (ox, oy, ox + side, oy + side), (max(1, dw), max(1, dh)), {'crop_jittered': jittered, 'center_crop_assumed': True}
    sx = dw / max(1, ow)
    sy = dh / max(1, oh)
    return None, (max(1, round(width * sx)), max(1, round(height * sy))), {'scale_x': sx, 'scale_y': sy, 'center_crop_assumed': False}


class PlatformAwareOnlineImageAugmentor:
    def __init__(self, config: Optional[RuntimeConfig] = None):
        self.cfg = config or RuntimeConfig()

    def set_epoch(self, epoch: int):
        """Call once per epoch for a different deterministic augmentation draw.

        When DataLoader uses persistent_workers=True, propagate epoch changes to
        the actual worker copies (or recreate the loader), not just this object.
        """
        self.cfg.epoch = int(epoch)

    def __call__(self, image, *, key: str, mask=None,
                 force_platform: Optional[str] = None,
                 force_quality: Optional[str] = None,
                 copy_index: int = 0,
                 return_info: bool = False):
        from PIL import Image
        im = image.copy().convert('RGB')
        sample = sample_policy('image', im.width, im.height, key=key, base_seed=self.cfg.seed, epoch=self.cfg.epoch, copy_index=copy_index, force_platform=force_platform, force_quality=force_quality, mixture=self.cfg.mixture())
        platform, recipe, rng = sample['platform'], sample['recipe'], sample['rng']
        info = {'mode': 'online-image', 'platform': platform, 'quality': sample['quality'], 'band': sample['band'], 'orientation': sample['orientation']}
        out_mask = mask.copy() if mask is not None else None
        if platform == 'clean':
            info['execution'] = 'identity'
            return (im, out_mask, info) if return_info else (im, out_mask)
        if recipe.get('reencode_verdict') == 'container_or_metadata_only':
            info['execution'] = 'identity'
            info['recipe_id'] = recipe['id']
            return (im, out_mask, info) if return_info else (im, out_mask)
        crop_box, target, geo = image_geometry(im.width, im.height, recipe, platform, rng, self.cfg)
        if crop_box is not None:
            im = im.crop(crop_box)
            if out_mask is not None:
                out_mask = out_mask.crop(crop_box)
        if im.size != target:
            im = im.resize(target, Image.Resampling.LANCZOS)
            if out_mask is not None:
                out_mask = out_mask.resize(target, Image.Resampling.NEAREST)
        photo = choose_photo(recipe, self.cfg.p_photo, self.cfg.photo_jitter, rng)
        if photo['applied']:
            im = adjust_luma_pillow(im, photo['a'], photo['b'])
        qf = inum(recipe.get('down_estimated_jpeg_qf'), 95) or 95
        qf = int(round(clamp(qf + rng.uniform(-self.cfg.jpeg_qf_jitter, self.cfg.jpeg_qf_jitter), 1, 100)))
        subsamp = recipe.get('down_chroma_subsampling', '4:2:0')
        subsampling = {'4:4:4': 0, '4:2:2': 1, '4:2:0': 2}.get(subsamp, 2)
        progressive = flag(recipe.get('down_jpeg_progressive'))
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=qf, subsampling=subsampling, progressive=progressive)
        buf.seek(0)
        with Image.open(buf) as decoded:
            out = decoded.convert('RGB').copy()
        info.update({'execution': 'jpeg-reencode', 'recipe_id': recipe['id'], 'recipe_source_row_ordinal': recipe.get('source_row_ordinal'), 'geometry': geo, 'photo': photo, 'jpeg': {'qf': qf, 'subsampling': subsamp, 'progressive': progressive}})
        return (out, out_mask, info) if return_info else (out, out_mask)


class PlatformAwareOnlineVideoAugmentor:
    def __init__(self, config: Optional[RuntimeConfig] = None):
        self.cfg = config or RuntimeConfig()

    def set_epoch(self, epoch: int):
        self.cfg.epoch = int(epoch)

    def augment_file(self, src: Path | str, dst: Optional[Path | str] = None,
                     *, key: Optional[str] = None,
                     force_platform: Optional[str] = None,
                     force_quality: Optional[str] = None,
                     copy_index: int = 0,
                     return_info: bool = False):
        src = Path(src)
        probe = ffprobe_json(src)
        sample = sample_policy('video', probe['width'], probe['height'], key=key or src.as_posix(), base_seed=self.cfg.seed, epoch=self.cfg.epoch, copy_index=copy_index, force_platform=force_platform, force_quality=force_quality, mixture=self.cfg.mixture())
        platform, recipe, rng = sample['platform'], sample['recipe'], sample['rng']
        info = {'mode': 'online-video', 'platform': platform, 'quality': sample['quality'], 'band': sample['band'], 'orientation': sample['orientation']}
        if dst is None:
            fd, tmp = tempfile.mkstemp(prefix='platform_aug_', suffix='.mp4')
            os.close(fd)
            dst = Path(tmp)
        else:
            dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if platform == 'clean' or (recipe is not None and recipe.get('reencode_verdict') == 'container_or_metadata_only'):
            shutil.copy2(src, dst)
            info['execution'] = 'identity'
            if recipe is not None:
                info['recipe_id'] = recipe['id']
            return (dst, info) if return_info else dst
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
        photo = choose_photo(recipe, self.cfg.p_photo, self.cfg.photo_jitter, rng)
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
        codec_args = ['-c:v', 'libx265' if is_hevc else 'libx264', '-preset', self.cfg.video_preset, '-pix_fmt', 'yuv420p']
        gop = max(1, inum(recipe.get('down_gop_mean'), 150) or 150)
        bframes = 2 if flag(recipe.get('down_has_b_frames')) else 0
        codec_args.extend(['-g', str(gop), '-keyint_min', str(gop), '-sc_threshold', '0', '-bf', str(bframes)])
        profile = str(recipe.get('down_profile', '')).lower()
        bitrate_target_bps = None
        if is_hevc:
            ratio = fnum(recipe.get('bitrate_ratio'), 0.3) or 0.3
            ratio *= rng.uniform(1 - self.cfg.hevc_bitrate_jitter, 1 + self.cfg.hevc_bitrate_jitter)
            input_bitrate = probe['bitrate_bps']
            if not input_bitrate:
                raise ValueError('HEVC online augmentation requires input bitrate from ffprobe')
            bitrate_target_bps = max(100_000, int(input_bitrate * ratio))
            codec_args.extend(['-b:v', str(bitrate_target_bps), '-tag:v', 'hvc1'])
            if profile in ('main', 'main10'):
                codec_args.extend(['-profile:v', 'main'])
        else:
            ref_crf = fnum(recipe.get('effective_crf_interp'), 26.0) or 26.0
            crf = clamp(ref_crf + rng.uniform(-self.cfg.h264_crf_jitter, self.cfg.h264_crf_jitter), 0, 51)
            codec_args.extend(['-crf', f'{crf:.2f}'])
            if profile in ('high', 'main', 'baseline', 'constrained baseline'):
                codec_args.extend(['-profile:v', 'baseline' if profile == 'constrained baseline' else profile])
        cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(src), '-map', '0:v:0', '-map', '0:a?']
        if filters:
            cmd += ['-vf', ','.join(filters)]
        cmd += codec_args + ['-r', f'{out_fps:.8g}', '-fps_mode', 'cfr', '-color_range', 'tv', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', str(dst)]
        run(cmd)
        info.update({'execution': 'video-transcode', 'recipe_id': recipe['id'], 'recipe_source_row_ordinal': recipe.get('source_row_ordinal'), 'geometry': geometry, 'photo': photo, 'video': {'codec': codec, 'profile': profile, 'out_fps': out_fps, 'gop': gop, 'bframes': bframes, 'bitrate_target_bps': bitrate_target_bps, 'effective_crf_reference': fnum(recipe.get('effective_crf_interp'), None)}})
        return (dst, info) if return_info else dst


class OnlineImageDatasetWrapper:
    """Wrap a base dataset and apply online image augmentation in __getitem__."""
    def __init__(self, base_dataset, augmentor: PlatformAwareOnlineImageAugmentor, key_fn: Optional[Callable[[Any, int], str]] = None, return_info: bool = False):
        self.base_dataset = base_dataset
        self.augmentor = augmentor
        self.key_fn = key_fn
        self.return_info = return_info

    def __len__(self):
        return len(self.base_dataset)

    def set_epoch(self, epoch: int):
        self.augmentor.set_epoch(epoch)

    def _parse(self, item, idx):
        if isinstance(item, dict):
            return item['image'], item.get('label'), item.get('key') or (self.key_fn(item, idx) if self.key_fn else f'idx:{idx}'), item.get('mask'), item
        if isinstance(item, tuple):
            if len(item) == 2:
                img, label = item
                return img, label, (self.key_fn(item, idx) if self.key_fn else f'idx:{idx}'), None, None
            if len(item) == 3:
                img, label, key = item
                return img, label, key, None, None
            if len(item) == 4:
                img, label, key, mask = item
                return img, label, key, mask, None
        raise TypeError('Unsupported dataset item format')

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        image, label, key, mask, original = self._parse(item, idx)
        out_img, out_mask, info = self.augmentor(image, key=key, mask=mask, return_info=True)
        if isinstance(item, dict):
            result = dict(original)
            result['image'] = out_img
            if out_mask is not None:
                result['mask'] = out_mask
            if self.return_info:
                result['aug_info'] = info
            return result
        if self.return_info:
            return out_img, label, key, out_mask, info
        return out_img, label


def parse_args():
    ap = argparse.ArgumentParser(description='Online platform-aware augmentation demos')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('demo-image', 'demo-video'):
        p = sub.add_parser(name)
        p.add_argument('--input', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--platform', type=str)
        p.add_argument('--quality', type=str)
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--epoch', type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = RuntimeConfig(seed=args.seed, epoch=args.epoch)
    if args.cmd == 'demo-image':
        from PIL import Image
        aug = PlatformAwareOnlineImageAugmentor(cfg)
        with Image.open(args.input) as im:
            out, _, info = aug(im, key=args.input.as_posix(), force_platform=args.platform, force_quality=args.quality, return_info=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.save(args.output)
        print(json.dumps(info, ensure_ascii=False, indent=2))
    elif args.cmd == 'demo-video':
        aug = PlatformAwareOnlineVideoAugmentor(cfg)
        _, info = aug.augment_file(args.input, args.output, key=args.input.as_posix(), force_platform=args.platform, force_quality=args.quality, return_info=True)
        print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()