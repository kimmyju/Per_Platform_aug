"""Optional per-file input metadata and detailed augmentation output metadata.

Main v2 stays CSV-free for its *empirical transformation policy*. This module
only reads a user-supplied CSV/JSON/JSONL of labels or sample attributes, when
requested, and joins it to each generated augmentation record.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath
from typing import Any, Optional


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def match_key(value: Any, mode: str) -> str:
    if value is None or str(value).strip() == "":
        raise ValueError("Input metadata join key cannot be empty")
    raw = str(value).strip().replace("\\", "/")
    if mode == "relative_path":
        while raw.startswith("./"):
            raw = raw[2:]
        result = str(PurePosixPath(raw))
    elif mode == "basename":
        result = PurePosixPath(raw).name
    elif mode == "stem":
        result = PurePosixPath(raw).stem
    else:
        raise ValueError(f"Unknown join mode: {mode}")
    if not result or result == ".":
        raise ValueError(f"Invalid metadata join key: {value!r}")
    return result


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input metadata file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"Empty CSV or missing header: {path}")
            return list(reader)
    if ext == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as stream:
            rows = []
            for n, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {n}: {path}") from exc
            return rows
    if ext == ".json":
        with path.open("r", encoding="utf-8-sig") as stream:
            obj = json.load(stream)
        if isinstance(obj, dict) and isinstance(obj.get("records"), list):
            obj = obj["records"]
        if not isinstance(obj, list):
            raise ValueError("Input JSON metadata must be a list or {'records': [...]} ")
        return obj
    raise ValueError("Input metadata supports .csv, .jsonl, or .json")


class InputMetadataIndex:
    def __init__(self, path: str | Path, *, key_field: str = "source_relative_path",
                 match_mode: str = "relative_path"):
        self.path = Path(path)
        self.key_field = key_field
        self.match_mode = match_mode
        self.rows: dict[str, dict[str, Any]] = {}
        for n, row in enumerate(_load_records(self.path), 1):
            if not isinstance(row, dict):
                raise ValueError(f"Input metadata record {n} is not an object")
            if key_field not in row:
                raise ValueError(
                    f"Missing metadata key field {key_field!r} in record {n}; "
                    "use --input-metadata-key to select its field")
            key = match_key(row[key_field], match_mode)
            if key in self.rows:
                raise ValueError(
                    f"Duplicate input metadata join key {key!r}; "
                    "use relative_path or a unique column to avoid ambiguity")
            self.rows[key] = row

    def lookup(self, relative_path: Path, *, allow_missing: bool = False
               ) -> Optional[dict[str, Any]]:
        key = match_key(relative_path.as_posix(), self.match_mode)
        row = self.rows.get(key)
        if row is None and not allow_missing:
            raise ValueError(f"No input metadata for {relative_path.as_posix()} (join key {key!r})")
        return row

    def validate_inputs(self, relative_paths: list[Path], *,
                        allow_missing: bool = False) -> None:
        seen: set[str] = set()
        missing: list[str] = []
        for rel in relative_paths:
            key = match_key(rel.as_posix(), self.match_mode)
            if key in seen:
                raise ValueError(
                    f"Two input media map to {key!r} with match mode "
                    f"{self.match_mode!r}; select --metadata-match-mode relative_path")
            seen.add(key)
            if key not in self.rows:
                missing.append(rel.as_posix())
        if missing and not allow_missing:
            raise ValueError(
                f"{len(missing)} input media lack input metadata. "
                f"First missing: {missing[:5]!r}. "
                "Provide matching keys or --allow-missing-input-metadata.")


CSV_FIELDS = [
    "source_relative_path", "input", "output", "status", "error",
    "copy_index", "seed", "effective_sample_seed", "epoch",
    "modality", "input_band", "input_size",
    "platform", "quality", "requested_condition", "selected_condition",
    "selection_method", "is_extrapolated", "aspect_log_distance",
    "size_log_distance", "selection_candidates",
    "reference_row_ordinal",
    "measured_reference_stimulus_id", "measured_reference_verdict",
    "operation", "crop_box", "resize_to", "jpeg_qf", "jpeg_subsampling",
    "jpeg_progressive", "brightness_b", "contrast_a", "photo_applied",
    "video_codec", "video_fps", "h264_crf", "hevc_target_bitrate_bps",
    "video_gop", "video_bframes", "ffmpeg_filter", "input_metadata",
    "measured_recipe", "augmentation_config", "transformation",
    "record_json",
]


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    transformation = record.get("transformation")
    transform = transformation if isinstance(transformation, dict) else {}
    geo = transform.get("geometry")
    geo = geo if isinstance(geo, dict) else {}
    photo = transform.get("photo")
    photo = photo if isinstance(photo, dict) else {}
    jpeg = transform.get("jpeg")
    jpeg = jpeg if isinstance(jpeg, dict) else {}
    video = transform.get("video")
    video = video if isinstance(video, dict) else {}
    comp: dict[str, Any] = dict(record)
    comp.update({
        "operation": transform.get("pixel_transform") or transform.get("verdict")
                     or (transformation if isinstance(transformation, str) else None),
        "crop_box": geo.get("crop_box"),
        "resize_to": geo.get("resize_to", geo.get("output_size")),
        "jpeg_qf": jpeg.get("qf"),
        "jpeg_subsampling": jpeg.get("subsampling"),
        "jpeg_progressive": jpeg.get("progressive"),
        "brightness_b": photo.get("b"),
        "contrast_a": photo.get("a"),
        "photo_applied": photo.get("applied"),
        "video_codec": video.get("codec"),
        "video_fps": video.get("fps"),
        "h264_crf": video.get("h264_crf"),
        "hevc_target_bitrate_bps": video.get("hevc_target_bitrate_bps"),
        "video_gop": video.get("reference_gop"),
        "video_bframes": video.get("bframes"),
        "ffmpeg_filter": video.get("ffmpeg_filter"),
        "record_json": _json(record),
    })
    flat: dict[str, Any] = {}
    for k in CSV_FIELDS:
        value = comp.get(k)
        flat[k] = _json(value) if isinstance(value, (list, dict)) else value
    return flat


class MetadataWriter:
    """Write one metadata entry per augmentation, with customizable path.

    .jsonl keeps native nested values. .csv additionally exposes important
    flat columns while preserving the complete original record in record_json.
    """
    def __init__(self, path: Path, *, append: bool = False):
        self.path = Path(path)
        self.append = append
        self.format = self.path.suffix.lower()
        if self.format not in (".jsonl", ".csv"):
            raise ValueError("--metadata-output must end in .jsonl or .csv")
        self.stream = None
        self.csv_writer = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        appending = self.append and self.path.is_file() and self.path.stat().st_size > 0
        self.stream = self.path.open("a" if appending else "w", encoding="utf-8", newline="")
        if self.format == ".csv":
            self.csv_writer = csv.DictWriter(
                self.stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if not appending:
                self.csv_writer.writeheader()
        return self

    def write(self, record: dict[str, Any]) -> None:
        if self.stream is None:
            raise RuntimeError("MetadataWriter must be opened with 'with'")
        if self.csv_writer is None:
            self.stream.write(_json(record) + "\n")
        else:
            self.csv_writer.writerow(_flatten_record(record))
        self.stream.flush()

    def __exit__(self, exc_type, exc, tb):
        if self.stream is not None:
            self.stream.close()
        return False
