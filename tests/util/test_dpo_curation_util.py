import tempfile
import unittest
from pathlib import Path

from modules.util.dpo_curation_util import export_curated_pairs

from PIL import Image


class DPOCurationUtilExportTest(unittest.TestCase):
    def test_export_pairs_copies_images_and_strips_angle_bracket_metadata_from_captions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            output_dir = temp_path / "output"
            source_dir.mkdir()

            chosen_path = source_dir / "chosen.png"
            rejected_path = source_dir / "rejected.jpg"

            Image.new("RGB", (8, 8), color="red").save(chosen_path)
            Image.new("RGB", (8, 8), color="blue").save(rejected_path)

            groups = [{
                "prompt": "portrait, <wildcard:hair>, cinematic light, <segment:face>, sharp focus",
                "aspectratio": "1:1",
                "images": [str(chosen_path), str(rejected_path)],
            }]
            results = {
                0: [
                    {
                        "chosen": str(chosen_path),
                        "rejected": str(rejected_path),
                    }
                ]
            }

            chosen_dir, rejected_dir, chosen_val_dir, rejected_val_dir, skipped, val_count, train_count = \
                export_curated_pairs(groups, results, str(output_dir), val_percentage=0.0)

            self.assertEqual(skipped, 0)

            exported_chosen_path = Path(chosen_dir) / "pair_0000_0000.png"
            exported_rejected_path = Path(rejected_dir) / "pair_0000_0000.jpg"
            exported_chosen_caption = Path(chosen_dir) / "pair_0000_0000.txt"
            exported_rejected_caption = Path(rejected_dir) / "pair_0000_0000.txt"

            self.assertTrue(exported_chosen_path.exists())
            self.assertTrue(exported_rejected_path.exists())
            self.assertEqual(exported_chosen_path.read_bytes(), chosen_path.read_bytes())
            self.assertEqual(exported_rejected_path.read_bytes(), rejected_path.read_bytes())

            expected_caption = "portrait, cinematic light, sharp focus"
            self.assertEqual(exported_chosen_caption.read_text(encoding="utf-8"), expected_caption)
            self.assertEqual(exported_rejected_caption.read_text(encoding="utf-8"), expected_caption)

    def test_has_existing_exports_detects_prior_output(self):
        from modules.util.dpo_curation_util import has_existing_exports

        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty directory — no prior exports
            self.assertFalse(has_existing_exports(temp_dir))

            # Create a chosen dir with a pair file
            chosen_dir = Path(temp_dir) / "chosen"
            chosen_dir.mkdir()
            (chosen_dir / "pair_0000.png").write_bytes(b"fake")
            self.assertTrue(has_existing_exports(temp_dir))

    def test_has_existing_exports_ignores_non_pair_files(self):
        from modules.util.dpo_curation_util import has_existing_exports

        with tempfile.TemporaryDirectory() as temp_dir:
            chosen_dir = Path(temp_dir) / "chosen"
            chosen_dir.mkdir()
            (chosen_dir / "something_else.png").write_bytes(b"fake")
            self.assertFalse(has_existing_exports(temp_dir))

    def test_export_single_pair_saves_immediately_and_updates_manifest(self):
        from modules.util.dpo_curation_util import (
            export_single_pair,
            load_manifest,
            manifest_pair_counts,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            output_dir = temp_path / "output"
            source_dir.mkdir()
            output_dir.mkdir()

            chosen_path = source_dir / "best.png"
            rejected_path = source_dir / "worst.jpg"
            Image.new("RGB", (8, 8), color="red").save(chosen_path)
            Image.new("RGB", (8, 8), color="blue").save(rejected_path)

            manifest = load_manifest(str(output_dir))
            self.assertEqual(manifest, {"pairs": []})

            export_single_pair(
                str(output_dir), manifest,
                str(chosen_path), str(rejected_path),
                "portrait, <wildcard:hair>, sharp focus",
                "1:1",
            )

            # Files exist on disk immediately
            self.assertTrue((output_dir / "chosen" / "pair_0000.png").exists())
            self.assertTrue((output_dir / "rejected" / "pair_0000.jpg").exists())

            # Caption has angle bracket segments stripped
            caption = (output_dir / "chosen" / "pair_0000.txt").read_text(encoding="utf-8")
            self.assertEqual(caption, "portrait, sharp focus")

            # Manifest is persisted
            reloaded = load_manifest(str(output_dir))
            self.assertEqual(len(reloaded["pairs"]), 1)
            self.assertEqual(reloaded["pairs"][0]["prompt"], "portrait, <wildcard:hair>, sharp focus")

            # Pair counts work — keyed by the normalized prompt so manifest
            # entries (which keep the raw prompt) and live group keys (which
            # are bracket-stripped during scan) both look up to the same slot.
            counts = manifest_pair_counts(reloaded)
            self.assertEqual(counts[("portrait, sharp focus", "1:1")], 1)

    def test_export_single_pair_increments_pair_id(self):
        from modules.util.dpo_curation_util import export_single_pair, load_manifest

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            output_dir = temp_path / "output"
            source_dir.mkdir()
            output_dir.mkdir()

            img_a = source_dir / "a.png"
            img_b = source_dir / "b.png"
            img_c = source_dir / "c.png"
            img_d = source_dir / "d.png"
            for p in (img_a, img_b, img_c, img_d):
                Image.new("RGB", (8, 8)).save(p)

            manifest = load_manifest(str(output_dir))
            export_single_pair(str(output_dir), manifest, str(img_a), str(img_b), "p1", "1:1")
            export_single_pair(str(output_dir), manifest, str(img_c), str(img_d), "p2", "16:9")

            self.assertEqual(len(manifest["pairs"]), 2)
            self.assertEqual(manifest["pairs"][0]["pair_id"], 0)
            self.assertEqual(manifest["pairs"][1]["pair_id"], 1)
            self.assertTrue((output_dir / "chosen" / "pair_0000.png").exists())
            self.assertTrue((output_dir / "chosen" / "pair_0001.png").exists())

    def test_finalize_export_creates_train_val_split(self):
        from modules.util.dpo_curation_util import (
            export_single_pair,
            finalize_export,
            load_manifest,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            output_dir = temp_path / "output"
            source_dir.mkdir()
            output_dir.mkdir()

            # Create 4 pairs
            manifest = load_manifest(str(output_dir))
            for i in range(4):
                chosen = source_dir / f"chosen_{i}.png"
                rejected = source_dir / f"rejected_{i}.png"
                Image.new("RGB", (8, 8), color="red").save(chosen)
                Image.new("RGB", (8, 8), color="blue").save(rejected)
                export_single_pair(
                    str(output_dir), manifest,
                    str(chosen), str(rejected),
                    f"prompt {i}", "1:1",
                )

            # All 4 pairs in flat chosen/rejected dirs
            self.assertEqual(len(list((output_dir / "chosen").glob("pair_*.png"))), 4)

            train_count, val_count = finalize_export(str(output_dir), manifest, val_percentage=50.0)

            self.assertEqual(train_count + val_count, 4)
            self.assertTrue(val_count >= 1)  # at least some go to val with 50%
            self.assertTrue((output_dir / "concepts.json").exists())
