"""Determinism/regression tests for standalone_online.py.

Run: python -m unittest -v test_standalone_online.py
"""
import csv
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import platform_aware_augment_standalone as core
from standalone_online import (
    CSVOnlineImageDataset,
    OnlineConfig,
    StandaloneOnlineAugmentor,
)


def rgb_digest(image):
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


class StableOnlineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "test.png"
        # Sample visually nonuniform pixels, so transforms can be compared.
        image = Image.new("RGB", (1080, 608))
        pix = image.load()
        for y in range(608):
            for x in range(1080):
                pix[x, y] = ((x // 7 + y) % 256,
                             (3*x + y // 3) % 256,
                             (x + 5*y) % 256)
        image.save(self.input)
        self.cfg = OnlineConfig(seed=42, epoch=0, temp_dir=self.tmp.name)
        self.aug = StandaloneOnlineAugmentor(self.cfg)

    def test_repeated_images_same_pixels_and_recipe(self):
        kwargs = dict(key="test/item-123", force_platform="facebook",
                      force_quality="low", return_info=True)
        im1, _, rec1 = self.aug.augment_image(self.input, **kwargs)
        im2, _, rec2 = self.aug.augment_image(self.input, **kwargs)
        self.assertEqual(rgb_digest(im1), rgb_digest(im2))
        self.assertEqual(rec1, rec2)

    def test_stable_across_call_order(self):
        keys = [f"source/item-{i:03d}" for i in range(12)]
        expected = {
            k: self.aug.plan_image(1080, 608, key=k,
                                   force_platform="facebook",
                                   force_quality="low")
            for k in keys
        }
        actual = {
            k: self.aug.plan_image(1080, 608, key=k,
                                   force_platform="facebook",
                                   force_quality="low")
            for k in reversed(keys)
        }
        self.assertEqual(expected, actual)

    def test_epoch_and_copy_index_in_sample_seed(self):
        a = self.aug.plan_image(1080, 608, key="a",
                                force_platform="facebook",
                                force_quality="low")
        b = self.aug.plan_image(1080, 608, key="a", copy_index=1,
                                force_platform="facebook",
                                force_quality="low")
        c = StandaloneOnlineAugmentor(
            self.cfg.with_epoch(1)
        ).plan_image(1080, 608, key="a",
                     force_platform="facebook", force_quality="low")
        self.assertNotEqual(a["sample_seed"], b["sample_seed"])
        self.assertNotEqual(a["sample_seed"], c["sample_seed"])

    def test_offline_rng_and_online_rng_choose_same_row(self):
        import random
        key = "dir/filename.jpg"
        record = self.aug.plan_image(
            1080, 608, key=key, force_platform="facebook",
            force_quality="low")
        rng = random.Random(core.stable_seed(42, Path(key), 0, 0))
        band = core.band_from_dimensions("image", 1080, 608)
        row, _ = core.MeasurementProfiles().choose(
            "facebook", "image", "low", band, 1080, 608, rng)
        self.assertEqual(record["reference_row_ordinal"],
                         row["reference_profile_row_ordinal"])

    def test_image_mask_tracks_geometry(self):
        mask_path = self.root / "mask.png"
        Image.new("L", (1080, 608), 1).save(mask_path)
        output, mask, info = self.aug.augment_image(
            self.input, key="image/a", mask_path=mask_path,
            force_platform="facebook", force_quality="low",
            return_info=True)
        self.assertEqual(output.size, mask.size)
        self.assertEqual(info["modality"], "image")

    def test_csv_shuffle_keeps_same_results(self):
        csv_path = self.root / "images.csv"
        def write_rows(ids):
            with csv_path.open("w", newline="", encoding="utf8") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "path", "label"])
                writer.writeheader()
                writer.writerows({"id": i, "path": self.input.name, "label": "fake"}
                                 for i in ids)
        ids = ["one", "two", "three"]
        write_rows(ids)
        original = CSVOnlineImageDataset(
            csv_path, self.root, self.aug, return_info=True)
        a = {original[i]["key"]: (
            rgb_digest(original[i]["image"]),
            original[i]["augmentation"]["sample_seed"])
             for i in range(len(original))}
        write_rows(ids[::-1])
        shuffled = CSVOnlineImageDataset(
            csv_path, self.root, self.aug, return_info=True)
        b = {shuffled[i]["key"]: (
            rgb_digest(shuffled[i]["image"]),
            shuffled[i]["augmentation"]["sample_seed"])
             for i in range(len(shuffled))}
        self.assertEqual(a, b)

    def test_duplicate_csv_ids_rejected(self):
        csv_path = self.root / "dups.csv"
        csv_path.write_text("id,path\nsame,test.png\nsame,test.png\n")
        with self.assertRaises(ValueError):
            CSVOnlineImageDataset(csv_path, self.root, self.aug)

    def test_video_tempfile_cleanup(self):
        video = self.root / "input.mp4"
        video.write_bytes(b"synthetic video placeholder")
        probe = {"width": 1080, "height": 608, "fps": 24.,
                 "bitrate_bps": 1000000}
        with patch.object(core, "ffprobe_json", return_value=probe):
            # Clean branch should reuse the original file.
            with self.aug.augment_video(
                video, key="video/001", force_platform="clean"
            ) as (path, info):
                self.assertEqual(path, video)
                self.assertTrue(path.exists())

            # Nonclean branch should own temporary output and clean it up.
            def fake_transcode(src, dst, row, platform, rng, args, video_probe):
                dst.write_bytes(b"deterministic transcoded bytes")
                return {"codec": "test"}
            with patch.object(core, "transform_video",
                              side_effect=fake_transcode):
                with self.aug.augment_video(
                    video, key="video/001", force_platform="facebook",
                    force_quality="low"
                ) as (tmp_path, info):
                    self.assertTrue(tmp_path.exists())
                    self.assertEqual(tmp_path.read_bytes(),
                                     b"deterministic transcoded bytes")
                self.assertFalse(tmp_path.exists())


    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "FFmpeg required for real video integration")
    def test_real_video_same_decoded_frames_on_repeated_calls(self):
        """Exercise the actual Standalone calibration + encode pipeline twice."""
        src = self.root / "real_video.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24",
            "-frames:v", "8", "-c:v", "libx264",
            "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src),
        ], check=True)
        def decode_hash(path):
            raw = subprocess.run([
                "ffmpeg", "-nostdin", "-loglevel", "error",
                "-i", str(path), "-an", "-map", "0:v:0",
                "-vf", "scale=160:160:flags=bilinear,format=gray",
                "-frames:v", "4", "-f", "rawvideo",
                "-pix_fmt", "gray", "pipe:1"
            ], check=True, capture_output=True).stdout
            return hashlib.sha256(raw).hexdigest()

        signatures = []
        metadata = []
        for _ in range(2):
            with self.aug.augment_video(
                src, key="video/real-001", force_platform="facebook",
                force_quality="low"
            ) as (output, info):
                signatures.append(decode_hash(output))
                metadata.append((info["sample_seed"],
                                 info["reference_row_ordinal"],
                                 info["platform"], info["quality"]))
                tmp_path = output
            self.assertFalse(tmp_path.exists())
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(metadata[0], metadata[1])

    def test_invalid_keys_rejected(self):
        for key in ("", " ", "/absolute/path", " whitespace "):
            with self.assertRaises(ValueError):
                self.aug.plan_image(1080, 608, key=key)


if __name__ == "__main__":
    unittest.main()
