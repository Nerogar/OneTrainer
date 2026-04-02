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
