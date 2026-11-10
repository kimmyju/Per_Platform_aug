#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replace a deterministic subset of *existing image augmentations* with video-codec
frame augmentations, starting from ORIGINAL frames listed in frames.csv.

Original source frames and the main augmentation script remain unchanged. A
deterministically selected subset of existing Perturb image files is REPLACED
IN PLACE after generating video-codec frames from the ORIGINAL frame path.
Both existing output images and metadata CSV are backed up before replacement.
The metadata CSV is atomically replaced, and previous settings remain audited.

A single frame is encoded as one I-picture. This approximates spatial video
codec artifacts ONLY; it cannot reproduce inter-frame prediction, GOP effects,
B-frame artifacts, motion compensation, video FPS, or frame offsets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from platform_aware_augment_standalone import (
    MeasurementProfiles, band_from_dimensions, clamp, fit_luma_ab,
    photo_residual, photo_targets, adjust_luma_pillow,
)
from standalone_metadata import CSV_FIELDS, _flatten_record


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-csv", required=True, type=Path)
    ap.add_argument("--metadata-csv", required=True, type=Path,
                    help="Existing Perturb_metadata.csv, kept unchanged")
    ap.add_argument("--images-root", required=True, type=Path,
                    help="Existing Perturb images/ root; output must be INSIDE this root")
    ap.add_argument("--backup-dir", required=True, type=Path,
                    help="NEW directory; backup original images and metadata here")
    ap.add_argument("--overwrite-existing-outputs", action="store_true",
                    help="Explicit consent to replace selected existing image files and metadata")
    ap.add_argument("--replace-fraction", type=float, default=1/3,
                    help="Fraction of eligible non-clean images to replace (default 1/3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fallback-policy", choices=["nearest", "strict"],
                    default="nearest")
    ap.add_argument("--hevc-crf", type=float, default=28,
                    help="Proposed x265 CRF for single-frame HEVC, not measured SNS CRF")
    ap.add_argument("--p-photo", type=float, default=1.0,
                    help="Probability of targeting measured video luminance a/b")
    ap.add_argument("--video-preset", choices=[
        "ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
        "slow", "slower", "veryslow"], default="medium")
    ap.add_argument("--plan-only", action="store_true",
                    help="Write a selection plan CSV only; do not alter existing files")
    ap.add_argument("--plan-output", type=Path,
                    help="Optional plan path; default Perturb_metadata_video_frame_plan.csv")
    return ap


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames:
            raise ValueError(f"No CSV header: {path}")
        return list(rd), list(rd.fieldnames)


def parse_json(value: Any, default: Any = None) -> Any:
    if not value:
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def keyed_random(seed: int, key: str, domain: str) -> int:
    payload = f"video-frame-postprocess-v1|{domain}|{seed}|{key}".encode("utf8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def is_eligible(row: dict[str, str]) -> bool:
    return row.get("status") == "ok" and row.get("platform") in (
        "facebook", "instagram", "kakao") and row.get("modality") == "image"


def video_quality(platform: str, image_quality: str) -> str:
    if platform == "facebook":
        if image_quality not in ("high", "mid", "low"):
            raise ValueError(f"Unknown Facebook image quality: {image_quality!r}")
        return image_quality
    if platform == "instagram":
        return "shared"
    if platform == "kakao":
        if image_quality not in ("original", "normal", "low"):
            raise ValueError(f"Unknown Kakao image quality: {image_quality!r}")
        # Kakao VIDEO has one measured file covering both H.264 and HEVC.
        # 'single' does NOT mean 'no compression' and is not an image tier.
        return "single"
    raise ValueError(f"Unknown platform: {platform!r}")


def sample_key(row: dict[str, str]) -> str:
    return row["source_relative_path"] + "|copy=" + str(row.get("copy_index", "0"))


def choose_subset(rows: list[dict[str, str]], fraction: float, seed: int) -> set[int]:
    """Exact overall count, approximately proportional class/category/platform strata."""
    eligible = [i for i, r in enumerate(rows) if is_eligible(r)]
    target = min(len(eligible), int(math.floor(len(eligible) * fraction + 0.5)))
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i in eligible:
        r = rows[i]
        source = parse_json(r.get("input_metadata"), {}) or {}
        group = (str(source.get("class", "")),
                 str(source.get("category", "")), r["platform"])
        groups[group].append(i)

    # Hamilton largest-remainder quotas: exactly target replacements.
    quotas = {key: int(math.floor(len(members) * fraction))
              for key, members in groups.items()}
    remain = target - sum(quotas.values())
    order = sorted(groups, key=lambda key: (
        -(len(groups[key]) * fraction - quotas[key]), key))
    for key in order[:remain]:
        quotas[key] += 1

    selected: set[int] = set()
    for group, members in groups.items():
        ranked = sorted(members, key=lambda i: (
            keyed_random(seed, sample_key(rows[i]), "selection"),
            sample_key(rows[i])))
        selected.update(ranked[:quotas[group]])
    if len(selected) != target:
        raise AssertionError(f"Sampling produced {len(selected)}, expected {target}")
    return selected


def even_dim(n: float) -> int:
    return max(2, 2 * int(round(n / 2)))


def geometric_frame(src: Path, row: dict[str, str], platform: str
                    ) -> tuple[Image.Image, dict[str, Any]]:
    with Image.open(src) as opened:
        im = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = im.size
    ow, oh = int(row["orig_width"]), int(row["orig_height"])
    dw, dh = int(row["down_width"]), int(row["down_height"])
    geometry: dict[str, Any] = {
        "input_size": [width, height], "reference_size": [ow, oh],
        "reference_download_size": [dw, dh], "center_crop_assumed": False,
        "crop_box": None, "resize_policy": "reference_ratio_isotropic_no_upscale",
    }

    if platform == "instagram":
        side = min(width, height)
        x = (width - side) // 2
        y = (height - side) // 2
        im = im.crop((x, y, x + side, y + side))
        geometry.update({"crop_box": [x, y, x + side, y + side],
                         "center_crop_assumed": True,
                         "resize_policy": "instagram_square_no_upscale"})
        # Reference video output is 720x720. Avoid inventing image detail by
        # upscaling a source whose short side is already smaller than 720.
        reference_side = min(dw, dh)
        target_side = even_dim(min(side, reference_side))
        target = (target_side, target_side)
    else:
        # Use a SINGLE scalar for both dimensions. A cross-orientation fallback
        # must never stretch faces independently in x/y.
        scale = min(1.0, dw / max(1, ow), dh / max(1, oh))
        target = (even_dim(width * scale), even_dim(height * scale))
        geometry["measured_scale_x"] = dw / max(1, ow)
        geometry["measured_scale_y"] = dh / max(1, oh)
        geometry["applied_isotropic_scale"] = scale

    if im.size != target:
        im = im.resize(target, Image.Resampling.LANCZOS)
    geometry["output_size"] = list(target)
    return im, geometry


def checked_run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            + result.stderr[-1600:])
    return result.stdout.strip()


def encode_decode(image: Image.Image, out_png: Path, codec: str, crf: float,
                  preset: str, work: Path, name: str) -> str:
    input_png = work / f"{name}_source.png"
    compressed = work / f"{name}.mp4"
    image.save(input_png, format="PNG")
    ff_codec = "libx265" if codec == "hevc" else "libx264"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-y", "-loop", "1", "-framerate", "1", "-i", str(input_png),
           "-frames:v", "1", "-an", "-c:v", ff_codec, "-preset", preset,
           "-pix_fmt", "yuv420p", "-g", "1", "-keyint_min", "1",
           "-sc_threshold", "0", "-bf", "0", "-crf", f"{crf:.3f}"]
    if codec == "hevc":
        cmd += ["-tag:v", "hvc1", "-x265-params", "pools=1:frame-threads=1"]
    checked_run(cmd + [str(compressed)])
    frame_type = checked_run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=pict_type", "-of", "csv=p=0", str(compressed)])
    picture_types = [line.strip().split(",")[0] for line in frame_type.splitlines()
                     if line.strip()]
    if picture_types != ["I"]:
        raise RuntimeError(f"Expected exactly one I-frame, got {frame_type!r}")
    checked_run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-y", "-i", str(compressed), "-frames:v", "1",
                 str(out_png)])
    return "I"


def video_frame_transform(src: Path, dst: Path, measured: dict[str, str],
                          platform: str, rng: random.Random,
                          options: argparse.Namespace) -> dict[str, Any]:
    codec = str(measured["down_codec_name"]).lower()
    codec = "hevc" if codec in ("hevc", "h265") else codec
    if codec not in ("hevc", "h264"):
        raise ValueError(f"Unsupported measured video codec {codec!r}")
    if codec == "h264":
        crf = float(measured.get("effective_crf_interp") or 26.0)
        crf_origin = "measured_x264_equivalent_crf"
    else:
        crf = options.hevc_crf
        crf_origin = "user_defined_x265_crf_NOT_empirically_measured"
    geometric, geometry = geometric_frame(src, measured, platform)
    photo_args = argparse.Namespace(p_photo=options.p_photo, photo_jitter=0.0)
    photo = photo_targets(measured, "video", rng, photo_args)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Staging output exists; refusing overwrite: {dst}")

    with tempfile.TemporaryDirectory(prefix="video_frame_", dir=dst.parent) as tmp:
        work = Path(tmp)
        baseline_file = work / "baseline.png"
        frame_type = encode_decode(
            geometric, baseline_file, codec, crf, options.video_preset, work, "baseline")
        with Image.open(baseline_file) as image:
            baseline_fit = fit_luma_ab(geometric, image.convert("RGB"))
        chosen = baseline_file
        photo_info: dict[str, Any] = {
            **photo, "codec_only_estimate": baseline_fit,
            "photometric_estimator": "Pillow 8-bit YCbCr OLS; approximate",
            "preencode_compensation": None,
        }
        if photo["applied"] and not baseline_fit.get("low_variance"):
            a0 = baseline_fit["a"]
            if abs(a0) > 0.05:
                a = clamp(photo["a"] / a0, 0.94, 1.06)
                b = clamp((photo["b"] - baseline_fit["b"]) / a0, -0.04, 0.04)
                photo_info["preencode_compensation"] = {"a": a, "b": b}
                if abs(a - 1.0) > 1e-7 or abs(b) > 1e-7:
                    corrected = adjust_luma_pillow(geometric, a, b)
                    trial = work / "trial.png"
                    encode_decode(corrected, trial, codec, crf,
                                  options.video_preset, work, "trial")
                    with Image.open(trial) as image:
                        trial_fit = fit_luma_ab(geometric, image.convert("RGB"))
                    photo_info["compensated_estimate"] = trial_fit
                    if photo_residual(trial_fit, photo) < photo_residual(
                            baseline_fit, photo):
                        chosen = trial
                        photo_info["compensation_accepted"] = True
                    else:
                        photo_info["compensation_accepted"] = False
        with Image.open(chosen) as image:
            final_fit = fit_luma_ab(geometric, image.convert("RGB"))
        photo_info["final_estimate"] = final_fit
        # Preserve the EXISTING output filename and extension, so downstream
        # directory scanners and CSV consumers need no changes. For .jpg
        # destinations, a high-quality JPEG delivery re-encode is unavoidable.
        # Record this explicitly as an extra lossy step, never claim pure HEVC.
        ext = dst.suffix.lower()
        if ext == ".png":
            with dst.open("xb") as out, chosen.open("rb") as inp:
                shutil.copyfileobj(inp, out)
            delivery = {"format": "PNG", "extra_jpeg_reencode": False}
        elif ext in (".jpg", ".jpeg"):
            with Image.open(chosen) as dec:
                dec_rgb = dec.convert("RGB")
                with dst.open("xb") as out:
                    dec_rgb.save(out, format="JPEG", quality=100, subsampling=0,
                                 progressive=False)
            delivery = {"format": "JPEG", "extra_jpeg_reencode": True,
                        "jpeg_qf": 100, "jpeg_subsampling": "4:4:4",
                        "note": "Additional high-quality JPEG save to preserve old .jpg filename"}
        else:
            raise ValueError(f"Unsupported destination extension: {dst}")

    return {
        "verdict": "single_frame_video_codec_spatial_approximation",
        "pixel_transform": "video_codec_frame",
        "geometry": geometry,
        "photo": photo_info,
        "delivery": delivery,
        "video": {
            "codec": codec, "codec_frame_type": frame_type,
            "h264_crf": round(crf, 3) if codec == "h264" else None,
            "hevc_crf": round(crf, 3) if codec == "hevc" else None,
            "crf_source": crf_origin,
            "pixel_format": "yuv420p", "output_file_format": delivery["format"],
            "intra_only": True, "fps_effect_applied": False,
            "gop_effect_applied": False, "bframe_effect_applied": False,
            "temporal_offset_applied": False,
            "reference_gop_not_applied": measured.get("down_gop_mean"),
            "reference_bframes_not_applied": measured.get("down_has_b_frames"),
            "reference_bitrate_ratio_not_applied": measured.get("bitrate_ratio"),
            "reference_fps_not_applied": measured.get("down_avg_frame_rate"),
            "approximation_limitation": "Single I-frame; no motion/temporal compression",
        },
    }


ADDED_FIELDS = [
    "augmentation_route", "source_modality", "recipe_modality",
    "postprocess_selected", "previous_output", "previous_platform",
    "previous_image_quality", "previous_image_operation",
    "previous_image_input_band", "previous_effective_sample_seed",
    "video_frame_seed", "codec_frame_type", "hevc_crf",
    "codec_quality_source", "postprocess_run", "output_replaced_in_place",
    "image_backup_path", "metadata_backup_path", "delivery_format",
    "delivery_jpeg_qf", "delivery_jpeg_subsampling",
]


def merged_record(original: dict[str, str], frame: dict[str, str] | None,
                  dst: Path | None, rng_seed: int | None,
                  options: argparse.Namespace, policy: MeasurementProfiles
                  ) -> dict[str, str]:
    updated = dict(original)
    previous_json = parse_json(original.get("record_json"), {}) or {}
    if not isinstance(previous_json, dict):
        raise ValueError("record_json should be a JSON object")
    source_meta = parse_json(original.get("input_metadata"), {}) or {}
    record = dict(previous_json)
    record["augmentation_route"] = (
        "video_codec_frame_overwrite" if dst is not None else
        "clean" if original.get("platform") == "clean" else "image")
    record["source_modality"] = "image"
    record["recipe_modality"] = "video" if dst is not None else (
        "image" if original.get("platform") != "clean" else None)
    record["postprocess_selected"] = dst is not None
    record["postprocess_run"] = {
        "method": "video_frame_postprocess_overwrite_v2", "selection_seed": options.seed,
        "replace_fraction_of_eligible": options.replace_fraction,
        "fallback_policy": options.fallback_policy,
    }
    updated.update({
        "augmentation_route": record["augmentation_route"],
        "source_modality": "image",
        "recipe_modality": record["recipe_modality"] or "",
        "postprocess_selected": str(dst is not None),
        "postprocess_run": json.dumps(record["postprocess_run"], sort_keys=True),
    })
    if dst is None:
        # Keep unselected record_json byte-for-byte identical; original settings
        # must not be accidentally rewritten or fabricated.
        return updated

    assert frame is not None and rng_seed is not None
    original_input = Path(frame["filepath"])
    if not original_input.is_file():
        raise FileNotFoundError(original_input)
    if original.get("input") != str(original_input):
        raise ValueError(
            f"Original input mismatch for {original['source_relative_path']!r}")
    platform = original["platform"]
    quality = video_quality(platform, original["quality"])
    with Image.open(original_input) as im:
        w, h = ImageOps.exif_transpose(im).size
    video_band = band_from_dimensions("video", w, h)
    rng = random.Random(rng_seed)
    measured, _ = policy.choose(
        platform, "video", quality, video_band, w, h, rng,
        fallback_policy=options.fallback_policy)
    # Stage on the same filesystem as the target and atomically swap only
    # after the video-codec image has been successfully generated.
    with tempfile.TemporaryDirectory(prefix=".vframe_stage_", dir=dst.parent) as stage_dir:
        staged = Path(stage_dir) / dst.name
        transform = video_frame_transform(original_input, staged, measured,
                                          platform, rng, options)
        backup_rel = dst.resolve().relative_to(options.images_root.resolve())
        backup = options.backup_dir / "images" / backup_rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            raise FileExistsError(f"Backup unexpectedly exists: {backup}")
        shutil.copy2(dst, backup)
        # Write a temporary copy, then atomically swap the stage into place.
        os.replace(staged, dst)
    options._replaced_files.append((dst, backup))

    previous = {
        "output": original.get("output"),
        "platform": original.get("platform"),
        "quality": original.get("quality"),
        "operation": original.get("operation"),
        "input_band": original.get("input_band"),
        "effective_sample_seed": original.get("effective_sample_seed"),
    }
    record.pop("conditional_quality_probabilities", None)
    record.update({
        "input": str(original_input), "output": str(dst),
        "status": "ok", "error": "", "modality": "image",
        "input_size": [w, h], "input_band": video_band,
        "platform": platform, "quality": quality,
        "effective_sample_seed": rng_seed,
        "requested_condition": measured["requested_condition"],
        "selected_condition": measured["selected_condition"],
        "selection_method": measured["selection_method"],
        "is_extrapolated": measured["is_extrapolated"],
        "aspect_log_distance": measured["aspect_log_distance"],
        "size_log_distance": measured["size_log_distance"],
        "selection_candidates": measured["selection_candidates"],
        "reference_row_ordinal": measured["reference_profile_row_ordinal"],
        "measured_reference_stimulus_id": measured["stimulus_id"],
        "measured_reference_verdict": measured["reencode_verdict"],
        "measured_reference_target_id": measured["target_id"],
        "reference_measurements_csv": "embedded_empirical_joint_observation_no_runtime_csv",
        "measured_recipe": measured, "transformation": transform,
        "input_metadata": source_meta, "previous_augmentation": previous,
        "conditional_quality_probabilities": {quality: 1.0} if platform != "facebook"
            else {"high": 0.5, "mid": 0.25, "low": 0.25},
        "output_replaced_in_place": True,
        "image_backup_path": str(backup),
        "metadata_backup_path": str(options._metadata_backup),
        "delivery_format": transform["delivery"]["format"],
        "delivery_jpeg_qf": transform["delivery"].get("jpeg_qf"),
        "delivery_jpeg_subsampling": transform["delivery"].get("jpeg_subsampling"),
        "previous_output": previous["output"], "previous_platform": previous["platform"],
        "previous_image_quality": previous["quality"],
        "previous_image_operation": previous["operation"],
        "previous_image_input_band": previous["input_band"],
        "previous_effective_sample_seed": previous["effective_sample_seed"],
        "video_frame_seed": rng_seed,
        "codec_frame_type": transform["video"]["codec_frame_type"],
        "hevc_crf": transform["video"]["hevc_crf"],
        "codec_quality_source": transform["video"]["crf_source"],
        "augmentation_config": {
            "type": "video_codec_frame_postprocess",
            "seed": options.seed, "video_frame_seed": rng_seed,
            "fallback_policy": options.fallback_policy,
            "hevc_crf": options.hevc_crf,
            "p_photo": options.p_photo,
            "video_preset": options.video_preset,
        },
    })
    updated.update(_flatten_record(record))
    # A JPG output necessarily involves a JPEG *delivery* encode after the
    # intra video codec. The standard jpeg_* fields must describe that step,
    # rather than staying empty or incorrectly representing the old recipe.
    if transform["delivery"]["extra_jpeg_reencode"]:
        updated["jpeg_qf"] = transform["delivery"]["jpeg_qf"]
        updated["jpeg_subsampling"] = transform["delivery"]["jpeg_subsampling"]
        updated["jpeg_progressive"] = False
        updated["record_json"] = json.dumps(record, ensure_ascii=False, sort_keys=True)
    for field in ADDED_FIELDS:
        value = record.get(field)
        updated[field] = (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list)) else value)
    return updated


def write_csv_atomic(dest: Path, fieldnames: list[str],
                     rows: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {dest}")
    with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=dest.parent,
            prefix=".new_metadata_", suffix=".csv", delete=False) as file:
        tmp = Path(file.name)
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    try:
        if overwrite:
            os.replace(tmp, dest)
        else:
            # Exclusive create ensures that an existing plan is never replaced.
            with dest.open("xb") as output, tmp.open("rb") as source:
                shutil.copyfileobj(source, output)
    finally:
        tmp.unlink(missing_ok=True)


def restore_replacements(entries: list[tuple[Path, Path]]) -> None:
    """Best-effort automatic rollback after any in-process failure."""
    failures = []
    for target, backup in reversed(entries):
        try:
            with tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=".restore_", suffix=target.suffix,
                    delete=False) as stream:
                temp = Path(stream.name)
            try:
                shutil.copy2(backup, temp)
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)
        except OSError as exc:
            failures.append((str(target), str(exc)))
    if failures:
        print("CRITICAL: rollback incomplete! Restore from --backup-dir:", failures,
              file=sys.stderr)


def main() -> None:
    args = parser().parse_args()
    if not 0 <= args.replace_fraction <= 1:
        raise ValueError("--replace-fraction must be between 0 and 1")
    if not 0 <= args.p_photo <= 1:
        raise ValueError("--p-photo must be between 0 and 1")
    if not 0 <= args.hevc_crf <= 51:
        raise ValueError("--hevc-crf must be between 0 and 51")
    old = args.metadata_csv.resolve()
    frames_src = args.frames_csv.resolve()
    image_root = args.images_root.resolve()
    backup_root = args.backup_dir.resolve()
    if old == frames_src:
        raise ValueError("frames.csv and Perturb metadata must differ")
    if not image_root.is_dir():
        raise ValueError(f"Missing existing images root: {image_root}")
    if backup_root == image_root or backup_root.is_relative_to(image_root):
        raise ValueError("Backup must be OUTSIDE the existing images root")
    if old.is_relative_to(backup_root) or frames_src.is_relative_to(backup_root):
        raise ValueError("Backup directory cannot contain the input CSVs")

    frame_rows, _ = load_csv(args.frames_csv)
    existing, original_fields = load_csv(args.metadata_csv)
    if not existing:
        raise ValueError("Existing metadata is empty")
    frame_by_rel = {}
    all_originals = set()
    for row in frame_rows:
        rel = row.get("relpath")
        if not rel or rel in frame_by_rel:
            raise ValueError(f"Blank or duplicate relpath in frames.csv: {rel!r}")
        if not row.get("filepath"):
            raise ValueError(f"Missing filepath for {rel}")
        frame_by_rel[rel] = row
        all_originals.add(Path(row["filepath"]).resolve())

    seen_keys, seen_outputs = set(), set()
    for row in existing:
        rel = row.get("source_relative_path")
        if not rel:
            raise ValueError("Missing source_relative_path in Perturb metadata")
        key = sample_key(row)
        if key in seen_keys:
            raise ValueError(f"Duplicate augmented frame key {key!r}")
        seen_keys.add(key)
        output = Path(row.get("output", "")).resolve()
        if output in seen_outputs:
            raise ValueError(f"Duplicate output path in Perturb metadata: {output}")
        seen_outputs.add(output)
        if is_eligible(row) and rel not in frame_by_rel:
            raise ValueError(f"No original frame corresponding to {rel}")

    # Do not silently re-process files already replaced by an earlier run.
    # A second run must use the untouched backup and the original metadata.
    already = [r["source_relative_path"] for r in existing
               if r.get("augmentation_route", "").startswith("video_codec_frame")
               or str(r.get("postprocess_selected", "")).lower() == "true"]
    if already:
        raise ValueError(
            f"Metadata already includes video-frame replacements ({len(already)} rows). "
            "Restore the original metadata and output backups before running again.")
    chosen = choose_subset(existing, args.replace_fraction, args.seed)
    targets: dict[int, Path] = {}
    for i in sorted(chosen):
        row = existing[i]
        dst = Path(row["output"])
        resolved = dst.resolve()
        if not resolved.is_relative_to(image_root):
            raise ValueError(f"Output is outside --images-root: {dst}")
        if dst.is_symlink() or not dst.is_file():
            raise ValueError(f"Output must be an existing regular file: {dst}")
        if resolved in all_originals or resolved == old or resolved == frames_src:
            raise ValueError(f"Refusing to overwrite an original source: {resolved}")
        if dst.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise ValueError(f"Unsupported existing image extension: {dst}")
        source = Path(frame_by_rel[row["source_relative_path"]]["filepath"])
        if not source.is_file():
            raise FileNotFoundError(f"Original frame missing: {source}")
        if row.get("input") != str(source):
            raise ValueError(f"Original frame mismatch for {row['source_relative_path']}")
        video_quality(row["platform"], row["quality"])
        targets[i] = dst
    print(f"Existing rows: {len(existing)}")
    print(f"Eligible non-clean image rows: {sum(is_eligible(x) for x in existing)}")
    print(f"Selected for ORIGINAL-frame video-codec replacement: {len(chosen)}")
    print(f"Unselected rows kept: {len(existing)-len(chosen)}")

    if args.plan_only:
        plan = args.plan_output or args.metadata_csv.with_name(
            "Perturb_metadata_video_frame_plan.csv")
        if plan.resolve() in (old, frames_src):
            raise ValueError("Plan output cannot overwrite original inputs")
        records = [{
            "source_relative_path": row["source_relative_path"],
            "original_file": frame_by_rel[row["source_relative_path"]]["filepath"],
            "overwrite_target": str(targets[i]),
            "platform": row["platform"],
            "previous_image_quality": row["quality"],
            "video_quality": video_quality(row["platform"], row["quality"]),
        } for i, row in enumerate(existing) if i in chosen]
        fields = ["source_relative_path", "original_file", "overwrite_target",
                  "platform", "previous_image_quality", "video_quality"]
        write_csv_atomic(plan, fields, records)
        print(f"Plan created (nothing overwritten): {plan}")
        return

    if not args.overwrite_existing_outputs:
        raise ValueError("Actual modification requires --overwrite-existing-outputs")
    if backup_root.exists():
        raise FileExistsError(f"Backup directory exists; choose a NEW path: {backup_root}")
    if chosen and (shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None):
        raise RuntimeError("ffmpeg and ffprobe are required for video-frame encoding")

    args._metadata_backup = backup_root / "metadata" / old.name
    args._replaced_files = []
    backup_root.mkdir(parents=True)
    args._metadata_backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old, args._metadata_backup)
    policy = MeasurementProfiles()
    out_rows = []
    try:
        for i, row in enumerate(existing):
            if i in chosen:
                rng_seed = keyed_random(args.seed, sample_key(row), "video_recipe")
                changed = merged_record(
                    row, frame_by_rel[row["source_relative_path"]],
                    targets[i], rng_seed, args, policy)
            else:
                changed = merged_record(row, None, None, None, args, policy)
            out_rows.append(changed)
            if i in chosen and (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(existing)} metadata rows", flush=True)
        fields = list(dict.fromkeys(original_fields + CSV_FIELDS + ADDED_FIELDS))
        # Source metadata is backed up, and replacement uses atomic rename.
        write_csv_atomic(old, fields, out_rows, overwrite=True)
    except BaseException:
        restore_replacements(args._replaced_files)
        # Metadata isn't replaced until all images succeed. Nonetheless restore
        # from its backup, including in the unlikely event of a write failure.
        if args._metadata_backup.is_file():
            with tempfile.NamedTemporaryFile(
                    dir=old.parent, prefix=".restore_metadata_", suffix=".csv",
                    delete=False) as stream:
                temp = Path(stream.name)
            try:
                shutil.copy2(args._metadata_backup, temp)
                os.replace(temp, old)
            finally:
                temp.unlink(missing_ok=True)
        raise
    print(f"Overwritten selected existing image files: {len(chosen)}")
    print(f"Metadata updated IN PLACE: {old}")
    print(f"Original frames unchanged: {frames_src}")
    print(f"Backup (original images + metadata): {backup_root}")
    print("NOTE: JPG destination adds an explicit QF100 4:4:4 JPEG save after video codec.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(2)
