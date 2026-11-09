"""frames.csv 앞 2장만으로 augmentation 이미지 생성과 메타데이터 저장을 확인한다.

입력 CSV:
  /media/NAS/DATASET/ProsecutorOffice/Perturb_Test/frames.csv

각 행의 filepath를 직접 읽고, 저장은 images/fake 또는 images/real 아래로 보낸다.
544x704(le512 portrait)에서는 Facebook 측정이 픽셀 보존이라,
이미지가 실제로 다시 만들어지는지 보려면 Instagram(shared)을 강제한다.

실행:
  python test_frames_csv_two.py
"""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "platform_aware_augment_standalone.py"
FRAMES_CSV = Path("/media/NAS/DATASET/ProsecutorOffice/Perturb_Test/frames.csv")
N_IMAGES = 2

META_FIELDS = (
    "relpath",
    "filepath",
    "label",
    "class",
    "category",
    "subgroup",
    "clip_id",
    "SKKU-D_prob_fake",
    "SKKU-E_prob_fake",
    "SKKU-G_prob_fake",
)


def load_first_rows(path: Path, n: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise AssertionError(f"헤더가 없습니다: {path}")
        missing = [name for name in META_FIELDS if name not in reader.fieldnames]
        if missing:
            raise AssertionError(f"CSV에 필요한 열이 없습니다: {missing}")
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(row)
            if len(rows) >= n:
                break
    if len(rows) < n:
        raise AssertionError(f"CSV 행이 {n}개보다 적습니다: {len(rows)}")
    return rows


class FramesCsvTwoImageTests(unittest.TestCase):
    def test_two_frames_are_written_with_metadata(self):
        if not FRAMES_CSV.is_file():
            self.skipTest(f"frames.csv를 찾을 수 없습니다: {FRAMES_CSV}")

        rows = load_first_rows(FRAMES_CSV, N_IMAGES)
        for row in rows:
            src = Path(row["filepath"])
            self.assertTrue(src.is_file(), f"원본 이미지가 없습니다: {src}")

        with tempfile.TemporaryDirectory(prefix="perturb_frames_two_") as tmp:
            root = Path(tmp)
            meta_csv = root / "input_meta.csv"
            with meta_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            output = root / "images"
            manifest = root / "Perturb_metadata.csv"
            cmd = [
                sys.executable, str(SCRIPT),
                "--input-list", str(meta_csv),
                "--split-by-class",
                "--output", str(output),
                "--copies", "1",
                "--seed", "42",
                "--epoch", "0",
                "--force-platform", "instagram",
                "--force-quality", "shared",
                "--metadata-output", str(manifest),
            ]
            run = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr + "\n" + run.stdout)

            with manifest.open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), N_IMAGES)
            self.assertEqual(
                [record["source_relative_path"] for record in records],
                [row["relpath"] for row in rows],
            )

            by_rel = {record["source_relative_path"]: record for record in records}
            for row in rows:
                record = by_rel[row["relpath"]]
                self.assertEqual(record["status"], "ok")
                self.assertEqual(record["platform"], "instagram")
                self.assertEqual(record["quality"], "shared")
                self.assertEqual(record["modality"], "image")
                self.assertEqual(record["copy_index"], "0")
                self.assertEqual(record["input"], row["filepath"])
                self.assertEqual(record["operation"], "resized_or_recompressed")
                saved = json.loads(record["input_metadata"])
                for field in META_FIELDS:
                    self.assertEqual(str(saved[field]), str(row[field]), field)

                out_path = Path(record["output"])
                self.assertEqual(out_path.relative_to(output).parts[0], "fake")
                self.assertTrue(out_path.is_file(), out_path)
                self.assertGreater(out_path.stat().st_size, 0)
                self.assertEqual(out_path.suffix.lower(), ".jpg")
                with Image.open(row["filepath"]) as source, Image.open(out_path) as image:
                    image.load()
                    self.assertEqual(image.format, "JPEG")
                    self.assertEqual(image.mode, "RGB")
                    self.assertGreater(image.width, 0)
                    self.assertGreater(image.height, 0)
                    # Instagram shared on this band is a center square, so the
                    # frame is no longer the original rectangle.
                    self.assertEqual(image.width, image.height)
                    self.assertNotEqual((image.width, image.height), source.size)

                transform = json.loads(record["transformation"])
                self.assertEqual(transform["verdict"], "resized_and_recompressed")
                self.assertIn("jpeg", transform)
                self.assertIn("geometry", transform)


if __name__ == "__main__":
    unittest.main()
