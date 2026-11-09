"""On-demand, per-sample deterministic access to the Standalone v2 policy.

The sampler and all image/video transformations are delegated to
platform_aware_augment_standalone.py.  This module never creates a durable
augmented dataset. Temporary files are placed under local temp_dir and removed.

Use a unique, immutable CSV sample ID as "key"; NEVER an iteration index,
worker ID, batch position, wall-clock time, or an absolute machine-specific path.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Optional

from PIL import Image, ImageOps

import platform_aware_augment_standalone as core


@dataclass(frozen=True)
class OnlineConfig:
    seed: int = 42
    epoch: int = 0
    p_clean: float = 0.25
    p_facebook: float = 0.25
    p_instagram: float = 0.25
    p_kakao: float = 0.25
    p_photo: float = 1.0
    photometric_sample_frames: int = 4
    photo_jitter: float = 0.0
    jpeg_qf_jitter: float = 0.0
    h264_crf_jitter: float = 0.0
    hevc_bitrate_jitter: float = 0.0
    ig_crop_jitter_prob: float = 0.0
    ig_crop_jitter_frac: float = 0.0
    video_preset: str = "medium"
    skip_video_output_probe: bool = False
    # Set to a LOCAL SSD or RAM tmpfs, never the full NAS.
    temp_dir: Optional[str] = None

    def __post_init__(self) -> None:
        weights = (self.p_clean, self.p_facebook,
                   self.p_instagram, self.p_kakao)
        if min(weights) < 0 or abs(sum(weights) - 1.0) > 1e-8:
            raise ValueError("Platform probabilities must be >=0 and sum to 1")
        if not (0 <= self.p_photo <= 1 and 0 <= self.ig_crop_jitter_prob <= 1):
            raise ValueError("Probabilities must be in [0,1]")
        if not (1 <= self.photometric_sample_frames <= 30):
            raise ValueError("photometric_sample_frames must be 1..30")
        for field in ("photo_jitter", "hevc_bitrate_jitter", "ig_crop_jitter_frac"):
            if not 0 <= getattr(self, field) <= 1:
                raise ValueError(f"{field} must be in [0,1]")
        if not 0 <= self.jpeg_qf_jitter <= 100:
            raise ValueError("jpeg_qf_jitter must be 0..100")
        if not 0 <= self.h264_crf_jitter <= 10:
            raise ValueError("h264_crf_jitter must be 0..10")
        if self.video_preset not in ("ultrafast", "superfast", "veryfast",
                                     "faster", "fast", "medium", "slow",
                                     "slower", "veryslow"):
            raise ValueError("Unsupported video_preset")

    def with_epoch(self, epoch: int) -> "OnlineConfig":
        """Return a NEW config. Keep epoch=0 for fixed model comparisons."""
        return replace(self, epoch=int(epoch))

    def fingerprint(self) -> str:
        """Hash every output-affecting option, excluding temp_dir."""
        values = asdict(self)
        values.pop("temp_dir")
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf8")).hexdigest()[:16]


def canonical_key(key: str) -> str:
    """Normalize slash separators but never derive an ID from dataset order."""
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Provide a nonempty, stable string sample key from CSV")
    if key != key.strip():
        raise ValueError("Sample key must not have leading/trailing whitespace")
    # Match core.stable_seed's rel.as_posix() for relative, POSIX CSV IDs.
    normalized = key.replace("\\", "/")
    if not normalized or normalized.startswith("/"):
        raise ValueError("Use a relative sample ID, not an absolute path")
    return PurePosixPath(normalized).as_posix()


class StandaloneOnlineAugmentor:
    """Stateless per-item sampler. Safe against shuffle/order/worker changes.

    Image augmentation uses the exact offline transform_image implementation
    with a short-lived LOCAL temporary file, decoded before the file is deleted.
    Video augmentation returns a context manager to ensure the transcoded
    temporary file is deleted after model inference.
    """

    def __init__(self, config: Optional[OnlineConfig] = None):
        self.config = config or OnlineConfig()
        self.policy = core.MeasurementProfiles()
        if self.config.temp_dir is not None:
            scratch = Path(self.config.temp_dir)
            if not scratch.is_dir():
                raise ValueError(f"temp_dir does not exist: {scratch}")

    def _args(self) -> Any:
        # Core transform functions use argparse.Namespace-like attribute access.
        from argparse import Namespace
        return Namespace(**asdict(self.config))

    def _draw(self, modality: str, width: int, height: int, *, key: str,
              copy_index: int = 0, force_platform: Optional[str] = None,
              force_quality: Optional[str] = None):
        key = canonical_key(key)
        if not isinstance(copy_index, int) or copy_index < 0:
            raise ValueError("copy_index must be a nonnegative integer")
        # Same hash and RNG consumption order as standalone's offline main().
        sampled_seed = core.stable_seed(
            self.config.seed, PurePosixPath(key), copy_index, self.config.epoch)
        rng = random.Random(sampled_seed)
        probs = [self.config.p_clean, self.config.p_facebook,
                 self.config.p_instagram, self.config.p_kakao]
        if force_platform is not None and force_platform not in core.PLATFORMS:
            raise ValueError(f"Unsupported platform: {force_platform}")
        platform = force_platform or rng.choices(core.PLATFORMS, probs, k=1)[0]
        band = core.band_from_dimensions(modality, width, height)
        record = {
            "sample_key": key,
            "base_seed": self.config.seed,
            "sample_seed": sampled_seed,
            "copy_index": copy_index,
            "epoch": self.config.epoch,
            "config_fingerprint": self.config.fingerprint(),
            "modality": modality,
            "input_size": [width, height],
            "band": band,
            "platform": platform,
        }
        if platform == "clean":
            if force_quality is not None:
                raise ValueError("Cannot force quality on clean route")
            record["quality"] = "none"
            record["reference_row_ordinal"] = None
            record["reference_stimulus_id"] = None
            return record, None, rng
        quality = core.pick_quality(platform, modality, rng, force_quality)
        row, _ = self.policy.choose(platform, modality, quality,
                                    band, width, height, rng)
        record.update({
            "quality": quality,
            "reference_stimulus_id": row["stimulus_id"],
            "reference_row_ordinal": row["reference_profile_row_ordinal"],
            "reencode_verdict": row["reencode_verdict"],
        })
        return record, row, rng

    def plan_image(self, width: int, height: int, *, key: str,
                   copy_index: int = 0, force_platform: Optional[str] = None,
                   force_quality: Optional[str] = None) -> dict:
        """JSON-serializable policy choice without encoding an image."""
        record, _, _ = self._draw(
            "image", width, height, key=key, copy_index=copy_index,
            force_platform=force_platform, force_quality=force_quality)
        return record

    def plan_video(self, path: str | Path, *, key: str, copy_index: int = 0,
                   force_platform: Optional[str] = None,
                   force_quality: Optional[str] = None) -> dict:
        """JSON-serializable video choice, probing dimensions only."""
        probe = core.ffprobe_json(Path(path))
        record, _, _ = self._draw(
            "video", probe["width"], probe["height"], key=key,
            copy_index=copy_index, force_platform=force_platform,
            force_quality=force_quality)
        return record

    def augment_image(self, path: str | Path, *, key: str,
                      copy_index: int = 0,
                      mask_path: Optional[str | Path] = None,
                      force_platform: Optional[str] = None,
                      force_quality: Optional[str] = None,
                      return_info: bool = False):
        """Load one image from CSV path and return PIL image (+ optional mask).

        No persistent output is produced. Uses the same transform_image as the
        standalone offline script, including JPEG calibration and mask geometry.
        """
        path = Path(path)
        with Image.open(path) as opened:
            display = ImageOps.exif_transpose(opened)
            width, height = display.size
        record, recipe, rng = self._draw(
            "image", width, height, key=key, copy_index=copy_index,
            force_platform=force_platform, force_quality=force_quality)
        mask_path = Path(mask_path) if mask_path is not None else None
        if mask_path is not None and not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if record["platform"] == "clean":
            with Image.open(path) as opened:
                output = ImageOps.exif_transpose(opened).convert("RGB").copy()
            mask = None
            if mask_path is not None:
                with Image.open(mask_path) as opened:
                    mask = opened.copy()
            record["transformation"] = "bytewise_copy_source_decoded_to_RGB"
        else:
            with tempfile.TemporaryDirectory(
                    prefix="standalone_img_", dir=self.config.temp_dir) as tmp:
                root = Path(tmp)
                verdict = recipe["reencode_verdict"]
                suffix = path.suffix if verdict == "container_or_metadata_only" else ".jpg"
                dest = root / ("result" + suffix)
                mask_dest = root / "mask.png" if mask_path is not None else None
                record["transformation"] = core.transform_image(
                    path, dest, recipe, record["platform"], rng,
                    self._args(), mask_path, mask_dest)
                with Image.open(dest) as opened:
                    output = ImageOps.exif_transpose(opened).convert("RGB").copy()
                mask = None
                if mask_dest is not None:
                    with Image.open(mask_dest) as opened:
                        mask = opened.copy()
        return (output, mask, record) if return_info else (output, mask)

    @contextmanager
    def augment_video(self, path: str | Path, *, key: str,
                      copy_index: int = 0,
                      force_platform: Optional[str] = None,
                      force_quality: Optional[str] = None
                      ) -> Iterator[tuple[Path, dict]]:
        """Yield (path, metadata) for one video; delete scratch on exit.

        IMPORTANT: Consume all video frames *inside* the with-block.
        For the clean route, returns the original file without touching it.
        """
        path = Path(path)
        probe = core.ffprobe_json(path)
        record, recipe, rng = self._draw(
            "video", probe["width"], probe["height"], key=key,
            copy_index=copy_index, force_platform=force_platform,
            force_quality=force_quality)
        if record["platform"] == "clean":
            record["transformation"] = "bytewise_copy_source_path"
            yield path, record
            return

        with tempfile.TemporaryDirectory(
                prefix="standalone_vid_", dir=self.config.temp_dir) as tmp:
            dest = Path(tmp) / (
                "result" + (path.suffix if
                    recipe["reencode_verdict"] == "container_or_metadata_only"
                    else ".mp4"))
            if recipe["reencode_verdict"] == "container_or_metadata_only":
                shutil.copy2(path, dest)
                record["transformation"] = "bytewise_copy"
            else:
                record["transformation"] = core.transform_video(
                    path, dest, recipe, record["platform"], rng,
                    self._args(), probe)
            yield dest, record


class CSVOnlineImageDataset:
    """CSV-backed lazy dataset; only reads a row's image when requested.

    CSV must have unique stable IDs and a path column. If using PyTorch's
    default DataLoader collate, supply a transform that converts PIL to Tensor.
    Never use CSV row index as augmentation key.
    """

    def __init__(self, csv_path: str | Path, image_root: str | Path,
                 augmentor: StandaloneOnlineAugmentor, *,
                 id_column: str = "id", path_column: str = "path",
                 label_column: Optional[str] = "label",
                 mask_column: Optional[str] = None,
                 transform: Optional[Callable[[Image.Image], Any]] = None,
                 return_info: bool = False, copy_index: int = 0):
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            required = [id_column, path_column]
            if any(field not in fields for field in required):
                raise ValueError(f"CSV needs columns {required}; got {fields}")
            if mask_column is not None and mask_column not in fields:
                raise ValueError(f"Missing mask column: {mask_column}")
            self.rows = list(reader)
        self.keys = [canonical_key(row[id_column]) for row in self.rows]
        if len(self.keys) != len(set(self.keys)):
            raise ValueError("Duplicate CSV IDs: use a unique stable ID per sample")
        self.root = Path(image_root)
        self.augmentor = augmentor
        self.id_column = id_column
        self.path_column = path_column
        self.label_column = label_column
        self.mask_column = mask_column
        self.transform = transform
        self.return_info = return_info
        self.copy_index = copy_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = Path(row[self.path_column])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        mask_path = None
        if self.mask_column and row.get(self.mask_column):
            mask_path = Path(row[self.mask_column])
            if not mask_path.is_absolute():
                mask_path = self.root / mask_path
        image, mask, info = self.augmentor.augment_image(
            image_path, key=self.keys[index], copy_index=self.copy_index,
            mask_path=mask_path, return_info=True)
        if self.transform is not None:
            image = self.transform(image)
        result = {"image": image, "key": self.keys[index]}
        if self.label_column is not None and self.label_column in row:
            result["label"] = row[self.label_column]
        if mask is not None:
            result["mask"] = mask
        if self.return_info:
            result["augmentation"] = info
        return result
