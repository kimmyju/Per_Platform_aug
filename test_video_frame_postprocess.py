"""End-to-end smoke tests for video-frame postprocessing."""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from postprocess_video_frame import video_quality, choose_subset

SCRIPT = Path(__file__).with_name("postprocess_video_frame.py")


class PostprocessTests(unittest.TestCase):
    def test_kakao_video_single_is_a_group_with_codec_settings(self):
        self.assertEqual(video_quality("kakao", "low"), "single")
        self.assertEqual(video_quality("kakao", "normal"), "single")
        self.assertEqual(video_quality("kakao", "original"), "single")

    def test_replacement_and_metadata_with_backed_up_in_place_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original_rows = []
            meta_rows = []
            for i, platform in enumerate(["facebook", "instagram", "kakao", "clean"]):
                rel = f"Fake/EFS/test/100/frame_{i:02d}.jpg"
                src = root / "original" / rel
                src.parent.mkdir(parents=True, exist_ok=True)
                size = (544, 704) if i != 2 else (1000, 1600)
                Image.new("RGB", size, (30 + i * 30, 80, 120)).save(src)
                # Include a WebP existing Perturb output to guard against
                # the exact unsupported-extension failure from real datasets.
                old_out = root / "old" / f"frame_{i:02d}{'.webp' if i == 0 else '.jpg'}"
                old_out.parent.mkdir(exist_ok=True)
                if i == 0:
                    with Image.open(src) as im:
                        im.save(old_out, format="WEBP", lossless=True)
                else:
                    shutil.copy2(src, old_out)
                original = dict(relpath=rel, filepath=str(src), class_="Fake",
                                label="1", category="EFS", clip_id="EFS/test/100")
                original["class"] = original.pop("class_")
                original_rows.append(original)
                meta = dict(source_relative_path=rel, input=str(src), output=str(old_out),
                            status="ok", modality="image", platform=platform,
                            quality={"facebook": "high", "instagram": "shared",
                                     "kakao": "low", "clean": ""}[platform],
                            copy_index="0", input_band="le512",
                            effective_sample_seed=str(i + 200),
                            operation="bytewise_copy", input_metadata=json.dumps(original))
                meta["record_json"] = json.dumps({**meta, "input_metadata": original,
                                                   "transformation": "bytewise_copy"})
                meta_rows.append(meta)
            frame_csv = root / "frames.csv"
            old_csv = root / "old.csv"
            img_dir = root / "old"
            backup_dir = root / "backups"
            for p, rows in [(frame_csv, original_rows), (old_csv, meta_rows)]:
                with p.open("w", newline="", encoding="utf8") as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            initial = [frame_csv.read_bytes(), old_csv.read_bytes()]
            old_images = [Path(row["output"]).read_bytes() for row in meta_rows]
            cmd = [sys.executable, str(SCRIPT), "--frames-csv", str(frame_csv),
                   "--metadata-csv", str(old_csv), "--images-root", str(img_dir),
                   "--backup-dir", str(backup_dir),
                   "--overwrite-existing-outputs",
                   "--seed", "42", "--replace-fraction", "1", "--p-photo", "0"]
            plan = subprocess.run(cmd + ["--plan-only"], capture_output=True, text=True)
            self.assertEqual(plan.returncode, 0, plan.stderr)
            self.assertEqual(old_csv.read_bytes(), initial[1])
            self.assertFalse(backup_dir.exists())
            if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
                self.skipTest("FFmpeg not installed")
            done = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
            with old_csv.open(newline="", encoding="utf8") as file:
                got = list(csv.DictReader(file))
            self.assertEqual(len(got), 4)
            self.assertEqual([r["augmentation_route"] for r in got],
                             ["video_codec_frame_overwrite"] * 3 + ["clean"])
            self.assertEqual(got[2]["video_codec"], "hevc")
            self.assertEqual(got[2]["quality"], "single")
            for i in range(3):
                self.assertEqual(got[i]["previous_output"], meta_rows[i]["output"])
                self.assertTrue(Path(got[i]["output"]).exists())
                self.assertEqual(got[i]["output"], meta_rows[i]["output"])
                expected_ext = ".webp" if i == 0 else ".jpg"
                self.assertEqual(Path(got[i]["output"]).suffix, expected_ext)
                with Image.open(got[i]["output"]) as image:
                    self.assertEqual(image.format, "WEBP" if i == 0 else "JPEG")
                if i == 0:
                    self.assertEqual(got[i]["delivery_format"], "WEBP")
                    self.assertEqual(got[i]["jpeg_qf"], "")
                    self.assertTrue(json.loads(got[i]["record_json"])
                                    ["transformation"]["delivery"]["lossless"])
                else:
                    self.assertEqual(got[i]["jpeg_qf"], "100")
                    self.assertEqual(got[i]["jpeg_subsampling"], "4:4:4")
                self.assertTrue(Path(got[i]["image_backup_path"]).exists())
                rec = json.loads(got[i]["record_json"])
                self.assertEqual(rec["transformation"]["video"]["codec_frame_type"], "I")
                self.assertEqual(rec["input_metadata"]["label"], "1")
            self.assertEqual(got[3]["output"], meta_rows[3]["output"])
            self.assertEqual(frame_csv.read_bytes(), initial[0])
            self.assertNotEqual(old_csv.read_bytes(), initial[1])
            self.assertEqual(
                [Path(got[i]["image_backup_path"]).read_bytes() for i in range(3)],
                old_images[:3])
            self.assertNotEqual([Path(row["output"]).read_bytes() for row in meta_rows][:3],
                                old_images[:3])
            self.assertEqual(Path(meta_rows[3]["output"]).read_bytes(), old_images[3])
            self.assertEqual(
                (backup_dir / "metadata" / "old.csv").read_bytes(), initial[1])
            repeated = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("Backup directory exists", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
