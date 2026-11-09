"""End-to-end tests for main_v2 Standalone offline metadata configuration."""
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


class MetadataIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        (self.inputs / "folder").mkdir(parents=True)
        self.image = self.inputs / "folder" / "picture.jpg"
        im = Image.new("RGB", (1080, 608), color=(125, 90, 66))
        im.save(self.image)
        self.meta_csv = self.root / "labels.csv"
        with self.meta_csv.open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "label", "split", "source_id"])
            writer.writeheader()
            writer.writerow({
                "filepath": "folder/picture.jpg",
                "label": "fake",
                "split": "test",
                "source_id": "demo-001",
            })

    def execute(self, *extra, output="result"):
        cmd = [
            sys.executable, str(SCRIPT),
            "--input", str(self.inputs),
            "--output", str(self.root / output),
            "--copies", "2",
            "--seed", "42",
            "--epoch", "0",
            "--force-platform", "facebook",
            "--force-quality", "low",
            *extra,
        ]
        return subprocess.run(cmd, text=True, capture_output=True)

    def test_user_csv_input_user_jsonl_output_and_determinism(self):
        path1 = self.root / "meta" / "first.jsonl"
        path2 = self.root / "meta" / "second.jsonl"
        options = [
            "--input-metadata", str(self.meta_csv),
            "--input-metadata-key", "filepath",
        ]
        a = self.execute(*options, "--metadata-output", str(path1), output="a")
        self.assertEqual(a.returncode, 0, a.stderr)
        b = self.execute(*options, "--metadata-output", str(path2), output="b")
        self.assertEqual(b.returncode, 0, b.stderr)
        one = [json.loads(x) for x in path1.read_text().splitlines()]
        two = [json.loads(x) for x in path2.read_text().splitlines()]
        self.assertEqual(len(one), 2)
        self.assertEqual([x["effective_sample_seed"] for x in one],
                         [x["effective_sample_seed"] for x in two])
        self.assertEqual([x["reference_row_ordinal"] for x in one],
                         [x["reference_row_ordinal"] for x in two])
        self.assertEqual([x["transformation"] for x in one],
                         [x["transformation"] for x in two])
        for record in one:
            self.assertEqual(record["input_metadata"]["label"], "fake")
            self.assertEqual(record["input_metadata"]["source_id"], "demo-001")
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["platform"], "facebook")
            self.assertEqual(record["quality"], "low")
            self.assertIn("measured_recipe", record)
            self.assertIn("augmentation_config", record)
            self.assertTrue(Path(record["output"]).exists())

    def test_user_jsonl_input_and_csv_output(self):
        source = self.root / "input.jsonl"
        source.write_text(json.dumps({
            "name": "picture.jpg", "label": "real", "patient": "X1"
        }) + "\n")
        output = self.root / "report" / "per_file.csv"
        run = self.execute(
            "--input-metadata", str(source),
            "--input-metadata-key", "name",
            "--metadata-match-mode", "basename",
            "--metadata-output", str(output),
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        with output.open(newline="", encoding="utf8") as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["platform"], "facebook")
            self.assertTrue(record["reference_row_ordinal"])
            self.assertEqual(json.loads(record["input_metadata"])["label"], "real")
            raw = json.loads(record["record_json"])
            self.assertEqual(raw["input_metadata"]["patient"], "X1")
            self.assertIn("measured_recipe", raw)
            self.assertIn("transformation", raw)

    def test_default_manifest_kept(self):
        run = self.execute()
        self.assertEqual(run.returncode, 0, run.stderr)
        p = self.root / "result" / "augmentation_manifest.jsonl"
        self.assertTrue(p.is_file())
        lines = [json.loads(x) for x in p.read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertIsNone(lines[0]["input_metadata"])

    def test_unmatched_input_metadata_fails_before_generation(self):
        self.meta_csv.write_text("filepath,label\nother.jpg,real\n")
        output = self.root / "meta.jsonl"
        run = self.execute(
            "--input-metadata", str(self.meta_csv),
            "--input-metadata-key", "filepath",
            "--metadata-output", str(output),
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(output.exists())
        self.assertIn("lack input metadata", run.stderr)

    def test_missing_allowed_explicitly(self):
        self.meta_csv.write_text("filepath,label\nother.jpg,real\n")
        output = self.root / "meta.jsonl"
        run = self.execute(
            "--input-metadata", str(self.meta_csv),
            "--input-metadata-key", "filepath",
            "--allow-missing-input-metadata",
            "--metadata-output", str(output),
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        row = json.loads(output.read_text().splitlines()[0])
        self.assertIsNone(row["input_metadata"])

    def test_metadata_output_cannot_overwrite_input(self):
        run = self.execute(
            "--input-metadata", str(self.meta_csv),
            "--input-metadata-key", "filepath",
            "--metadata-output", str(self.meta_csv),
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(
            len(list(csv.DictReader(self.meta_csv.open(encoding="utf8")))), 1)

    def test_duplicate_metadata_keys_rejected(self):
        self.meta_csv.write_text(
            "filepath,label\nfolder/picture.jpg,fake\nfolder/picture.jpg,real\n")
        run = self.execute(
            "--input-metadata", str(self.meta_csv),
            "--input-metadata-key", "filepath",
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Duplicate input metadata join key", run.stderr)

    def test_input_list_saves_under_real_and_fake(self):
        real = self.inputs / "Real" / "clip" / "a.jpg"
        fake = self.inputs / "Fake" / "clip" / "a.jpg"
        real.parent.mkdir(parents=True)
        fake.parent.mkdir(parents=True)
        Image.new("RGB", (1080, 608), color=(10, 20, 30)).save(real)
        Image.new("RGB", (544, 704), color=(30, 40, 50)).save(fake)
        listed = self.root / "frames.csv"
        with listed.open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["filepath", "relpath", "class", "label"])
            writer.writeheader()
            writer.writerow({
                "filepath": str(real), "relpath": "Real/clip/a.jpg",
                "class": "Real", "label": "0",
            })
            writer.writerow({
                "filepath": str(fake), "relpath": "Fake/clip/a.jpg",
                "class": "Fake", "label": "1",
            })
        meta = self.root / "Perturb_metadata.csv"
        run = subprocess.run([
            sys.executable, str(SCRIPT),
            "--input-list", str(listed),
            "--split-by-class",
            "--output", str(self.root / "images"),
            "--metadata-output", str(meta),
            "--copies", "1",
            "--seed", "42",
            "--force-platform", "facebook",
            "--force-quality", "low",
        ], text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        with meta.open(newline="", encoding="utf8") as handle:
            records = list(csv.DictReader(handle))
        by_class = {json.loads(row["input_metadata"])["class"]: row for row in records}
        self.assertEqual(len(by_class), 2)
        real_out = Path(by_class["Real"]["output"])
        fake_out = Path(by_class["Fake"]["output"])
        self.assertEqual(
            real_out.relative_to(self.root / "images").parts[:2], ("real", "clip"))
        self.assertEqual(
            fake_out.relative_to(self.root / "images").parts[:2], ("fake", "clip"))
        self.assertTrue(real_out.is_file())
        self.assertTrue(fake_out.is_file())
        self.assertEqual(by_class["Real"]["input"], str(real))
        self.assertEqual(by_class["Fake"]["platform"], "facebook")
        self.assertTrue(by_class["Fake"]["operation"])

    def test_unsupported_landscape_uses_fallback_with_csv_provenance(self):
        """Same sample/seed twice gives identical nearest measured row."""
        land = self.inputs / "folder" / "landscape.jpg"
        Image.new("RGB", (640, 480), color=(110, 20, 190)).save(land)
        listed = self.root / "landscape.csv"
        with listed.open("w", newline="", encoding="utf8") as fh:
            wr = csv.DictWriter(fh, fieldnames=["filepath", "relpath", "class"])
            wr.writeheader()
            wr.writerow({"filepath": str(land), "relpath": "Fake/landscape.jpg",
                         "class": "Fake"})
        reports = []
        for idx in range(2):
            output = self.root / f"fallback_{idx}.csv"
            cmd = [
                sys.executable, str(SCRIPT),
                "--input-list", str(listed), "--split-by-class",
                "--output", str(self.root / f"fallback_images_{idx}"),
                "--metadata-output", str(output),
                "--copies", "1", "--seed", "42", "--epoch", "0",
                "--force-platform", "kakao", "--force-quality", "low",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            with output.open(newline="", encoding="utf8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            rec = rows[0]
            self.assertEqual(rec["status"], "ok")
            self.assertEqual(rec["quality"], "low")
            self.assertEqual(rec["requested_condition"], "le512/land")
            self.assertEqual(rec["selected_condition"], "le512/port")
            self.assertEqual(rec["selection_method"], "nearest_same_band")
            self.assertEqual(rec["is_extrapolated"], "True")
            self.assertEqual(json.loads(rec["input_metadata"])["class"], "Fake")
            self.assertTrue(Path(rec["output"]).is_file())
            with Image.open(rec["output"]) as im:
                self.assertEqual(im.size, (640, 480))
            reports.append(rec)
        self.assertEqual(reports[0]["effective_sample_seed"],
                         reports[1]["effective_sample_seed"])
        self.assertEqual(reports[0]["reference_row_ordinal"],
                         reports[1]["reference_row_ordinal"])
        self.assertEqual(reports[0]["transformation"], reports[1]["transformation"])

    def test_unsupported_landscape_strict_rejects_and_logs(self):
        land = self.inputs / "folder" / "landscape.jpg"
        Image.new("RGB", (640, 480), color=(110, 20, 190)).save(land)
        listed = self.root / "landscape.csv"
        listed.write_text("filepath,relpath,class\n" + str(land) +
                          ",Fake/landscape.jpg,Fake\n", encoding="utf8")
        report = self.root / "strict.jsonl"
        cmd = [
            sys.executable, str(SCRIPT),
            "--input-list", str(listed), "--split-by-class",
            "--output", str(self.root / "strict_out"),
            "--metadata-output", str(report),
            "--copies", "1", "--seed", "42",
            "--force-platform", "kakao", "--force-quality", "low",
            "--fallback-policy", "strict",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        record = json.loads(report.read_text(encoding="utf8").splitlines()[0])
        self.assertEqual(record["status"], "error")
        self.assertIn("No measured profile", record["error"])
        self.assertEqual(record["quality"], "low")

    def test_filename_matching_ambiguity_is_rejected(self):
        second = self.inputs / "second"
        second.mkdir()
        Image.new("RGB", (1080, 608), color=(1, 2, 3)).save(second / "picture.jpg")
        run = self.execute(
            "--input-metadata", str(self.meta_csv),
            "--input-metadata-key", "filepath",
            "--metadata-match-mode", "basename",
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Two input media map", run.stderr)


if __name__ == "__main__":
    unittest.main()
