import os
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

            groups = [
                {
                    "prompt": "portrait, <wildcard:hair>, cinematic light, <segment:face>, sharp focus",
                    "aspectratio": "1:1",
                    "images": [str(chosen_path), str(rejected_path)],
                }
            ]
            results = {
                0: [
                    {
                        "chosen": str(chosen_path),
                        "rejected": str(rejected_path),
                    }
                ]
            }

            chosen_dir, rejected_dir, chosen_val_dir, rejected_val_dir, skipped, val_count, train_count = (
                export_curated_pairs(groups, results, str(output_dir), val_percentage=0.0)
            )

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
                str(output_dir),
                manifest,
                str(chosen_path),
                str(rejected_path),
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
                    str(output_dir),
                    manifest,
                    str(chosen),
                    str(rejected),
                    f"prompt {i}",
                    "1:1",
                )

            # All 4 pairs in flat chosen/rejected dirs
            self.assertEqual(len(list((output_dir / "chosen").glob("pair_*.png"))), 4)

            train_count, val_count = finalize_export(str(output_dir), manifest, val_percentage=50.0)

            self.assertEqual(train_count + val_count, 4)
            self.assertTrue(val_count >= 1)  # at least some go to val with 50%
            self.assertTrue((output_dir / "concepts.json").exists())


class DPOCaptionMismatchTest(unittest.TestCase):
    def _build_pair(self, root: Path, name: str, chosen_caption: str | None, rejected_caption: str | None):
        """Create chosen/rejected image pair with optional sidecar captions.
        None means 'no .txt sidecar'; '' means an empty sidecar file."""
        chosen_dir = root / "chosen"
        rejected_dir = root / "rejected"
        chosen_dir.mkdir(exist_ok=True)
        rejected_dir.mkdir(exist_ok=True)
        chosen_img = chosen_dir / f"{name}.png"
        rejected_img = rejected_dir / f"{name}.png"
        Image.new("RGB", (4, 4), color="red").save(chosen_img)
        Image.new("RGB", (4, 4), color="blue").save(rejected_img)
        if chosen_caption is not None:
            (chosen_dir / f"{name}.txt").write_text(chosen_caption, encoding="utf-8")
        if rejected_caption is not None:
            (rejected_dir / f"{name}.txt").write_text(rejected_caption, encoding="utf-8")
        return chosen_dir, rejected_dir, chosen_img, rejected_img

    def test_identical_captions_have_no_mismatches(self):
        from modules.util.dpo_curation_util import find_caption_mismatches

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chosen_dir, rejected_dir, _, _ = self._build_pair(root, "pair_0001", "a cat", "a cat")
            mismatches = find_caption_mismatches([(str(chosen_dir), str(rejected_dir))])
            self.assertEqual(mismatches, [])

    def test_trailing_newline_difference_is_not_a_mismatch(self):
        from modules.util.dpo_curation_util import find_caption_mismatches

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chosen_dir, rejected_dir, _, _ = self._build_pair(root, "pair_0001", "a cat", "a cat\n")
            mismatches = find_caption_mismatches([(str(chosen_dir), str(rejected_dir))])
            self.assertEqual(mismatches, [])

    def test_differing_captions_produce_mismatch_entry(self):
        from modules.util.dpo_curation_util import find_caption_mismatches

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chosen_dir, rejected_dir, chosen_img, rejected_img = self._build_pair(
                root,
                "pair_0001",
                "a cat",
                "a dog",
            )
            mismatches = find_caption_mismatches([(str(chosen_dir), str(rejected_dir))])
            self.assertEqual(len(mismatches), 1)
            entry = mismatches[0]
            self.assertEqual(entry["key"], "pair_0001")
            self.assertEqual(entry["chosen_image"], str(chosen_img))
            self.assertEqual(entry["rejected_image"], str(rejected_img))
            self.assertEqual(entry["chosen_caption"], "a cat")
            self.assertEqual(entry["rejected_caption"], "a dog")
            self.assertIsNotNone(entry["chosen_caption_path"])
            self.assertIsNotNone(entry["rejected_caption_path"])

    def test_missing_rejected_sidecar_counts_as_mismatch(self):
        from modules.util.dpo_curation_util import find_caption_mismatches

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chosen_dir, rejected_dir, _, _ = self._build_pair(root, "pair_0001", "a cat", None)
            mismatches = find_caption_mismatches([(str(chosen_dir), str(rejected_dir))])
            self.assertEqual(len(mismatches), 1)
            self.assertEqual(mismatches[0]["chosen_caption"], "a cat")
            self.assertEqual(mismatches[0]["rejected_caption"], "")
            self.assertIsNone(mismatches[0]["rejected_caption_path"])

    def test_unmatched_strays_are_not_in_caption_mismatches(self):
        from modules.util.dpo_curation_util import find_caption_mismatches

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chosen_dir = root / "chosen"
            rejected_dir = root / "rejected"
            chosen_dir.mkdir()
            rejected_dir.mkdir()
            # Chosen-only stray (no rejected counterpart)
            Image.new("RGB", (4, 4)).save(chosen_dir / "lonely.png")
            (chosen_dir / "lonely.txt").write_text("solo", encoding="utf-8")
            mismatches = find_caption_mismatches([(str(chosen_dir), str(rejected_dir))])
            self.assertEqual(mismatches, [])

    def test_apply_caption_to_pair_writes_both_sidecars(self):
        from modules.util.dpo_curation_util import apply_caption_to_pair

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, _, chosen_img, rejected_img = self._build_pair(root, "pair_0001", "old", "older")
            apply_caption_to_pair(str(chosen_img), str(rejected_img), "new caption")
            chosen_txt = chosen_img.with_suffix(".txt")
            rejected_txt = rejected_img.with_suffix(".txt")
            self.assertEqual(chosen_txt.read_text(encoding="utf-8"), "new caption")
            self.assertEqual(rejected_txt.read_text(encoding="utf-8"), "new caption")

    def test_apply_caption_to_pair_creates_missing_sidecar(self):
        from modules.util.dpo_curation_util import apply_caption_to_pair

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, _, chosen_img, rejected_img = self._build_pair(root, "pair_0001", "old", None)
            apply_caption_to_pair(str(chosen_img), str(rejected_img), "fresh")
            self.assertEqual(chosen_img.with_suffix(".txt").read_text(encoding="utf-8"), "fresh")
            self.assertEqual(rejected_img.with_suffix(".txt").read_text(encoding="utf-8"), "fresh")

    def test_correct_all_captions_to_chosen_overwrites_rejected(self):
        from modules.util.dpo_curation_util import (
            correct_all_captions_to_chosen,
            find_caption_mismatches,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chosen_dir, rejected_dir, chosen_img, rejected_img = self._build_pair(
                root,
                "pair_0001",
                "the truth",
                "garbled",
            )
            mismatches = find_caption_mismatches([(str(chosen_dir), str(rejected_dir))])
            corrected = correct_all_captions_to_chosen(mismatches)
            self.assertEqual(corrected, 1)
            self.assertEqual(rejected_img.with_suffix(".txt").read_text(encoding="utf-8"), "the truth")
            # Re-run detector: zero mismatches now (idempotent)
            self.assertEqual(find_caption_mismatches([(str(chosen_dir), str(rejected_dir))]), [])

    def test_correct_all_creates_rejected_sidecar_when_missing(self):
        from modules.util.dpo_curation_util import (
            correct_all_captions_to_chosen,
            find_caption_mismatches,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chosen_dir, rejected_dir, _, rejected_img = self._build_pair(
                root,
                "pair_0001",
                "canonical",
                None,
            )
            mismatches = find_caption_mismatches([(str(chosen_dir), str(rejected_dir))])
            self.assertEqual(correct_all_captions_to_chosen(mismatches), 1)
            self.assertEqual(rejected_img.with_suffix(".txt").read_text(encoding="utf-8"), "canonical")


class WalkSkippingDottedTest(unittest.TestCase):
    def test_dotted_subdirectories_are_pruned_at_any_depth(self):
        from modules.util.dpo_curation_util import walk_skipping_dotted

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "keep").mkdir()
            (root / ".thumbnails").mkdir()
            (root / "keep" / ".cache" / "deep").mkdir(parents=True)
            (root / "visible.png").touch()
            (root / "keep" / "nested.png").touch()
            (root / ".thumbnails" / "thumb.png").touch()
            (root / "keep" / ".cache" / "deep" / "preview.png").touch()

            found = [os.path.join(r, f) for r, files in walk_skipping_dotted(str(root)) for f in files]
            names = sorted(os.path.basename(p) for p in found)
            self.assertEqual(names, ["nested.png", "visible.png"])

    def test_explicitly_selected_dotted_root_is_still_walked(self):
        from modules.util.dpo_curation_util import walk_skipping_dotted

        with tempfile.TemporaryDirectory() as temp_dir:
            dotted_root = Path(temp_dir) / ".my_folder"
            dotted_root.mkdir()
            (dotted_root / "img.png").touch()
            found = [f for _r, files in walk_skipping_dotted(str(dotted_root)) for f in files]
            self.assertEqual(found, ["img.png"])


class TrainerBucketKeyTest(unittest.TestCase):
    """Grouping must mirror the trainer's AspectBucketing: images that crop to
    the same bucket belong in the same curation group, regardless of exact
    pixel dimensions or generator metadata strings."""

    def test_common_16_9_inference_resolutions_share_a_bucket(self):
        from modules.util.dpo_curation_util import trainer_bucket_key

        # Both marketed as 16:9 by inference frontends; true ratio 1.75 -> 7:4.
        self.assertEqual(trainer_bucket_key(1344, 768), "7:4")
        self.assertEqual(trainer_bucket_key(1680, 960), "7:4")

    def test_one_pixel_off_still_lands_in_same_bucket(self):
        from modules.util.dpo_curation_util import trainer_bucket_key

        self.assertEqual(trainer_bucket_key(1343, 768), trainer_bucket_key(1344, 768))
        self.assertEqual(trainer_bucket_key(1344, 769), trainer_bucket_key(1344, 768))

    def test_orientation_is_preserved(self):
        from modules.util.dpo_curation_util import trainer_bucket_key

        self.assertEqual(trainer_bucket_key(768, 1344), "4:7")
        self.assertEqual(trainer_bucket_key(1024, 1024), "1:1")

    def test_extreme_aspect_clamps_to_widest_bucket(self):
        from modules.util.dpo_curation_util import trainer_bucket_key

        self.assertEqual(trainer_bucket_key(5000, 1000), "4:1")
        self.assertEqual(trainer_bucket_key(1000, 5000), "1:4")

    def test_non_positive_dimensions_return_empty(self):
        from modules.util.dpo_curation_util import trainer_bucket_key

        self.assertEqual(trainer_bucket_key(0, 768), "")
        self.assertEqual(trainer_bucket_key(1344, 0), "")

    def test_normalize_aspect_maps_metadata_strings_to_buckets(self):
        from modules.util.dpo_curation_util import normalize_aspect_for_grouping

        # SwarmUI writes "16:9" while actually generating 1.75 images.
        self.assertEqual(normalize_aspect_for_grouping("16:9"), "7:4")
        self.assertEqual(normalize_aspect_for_grouping("1:1"), "1:1")
        self.assertEqual(normalize_aspect_for_grouping("1343:768"), "7:4")
        self.assertEqual(normalize_aspect_for_grouping("1.78"), "7:4")
        # Unparseable strings pass through so legacy keys never silently merge.
        self.assertEqual(normalize_aspect_for_grouping(""), "")
        self.assertEqual(normalize_aspect_for_grouping("portrait"), "portrait")

    def test_resolve_aspect_ratio_prefers_pixels_over_metadata(self):
        from modules.util.dpo_curation_util import resolve_aspect_ratio

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "img.png"
            Image.new("RGB", (1344, 768)).save(path)
            # Metadata says 16:9, pixels say 1.75 — both snap to 7:4, but the
            # pixel dimensions are the authority (the trainer never reads
            # metadata).
            self.assertEqual(resolve_aspect_ratio("16:9", str(path)), "7:4")
            self.assertEqual(resolve_aspect_ratio("", str(path)), "7:4")

    def test_resolve_aspect_ratio_falls_back_to_metadata_when_undecodable(self):
        from modules.util.dpo_curation_util import resolve_aspect_ratio

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.png"
            path.write_bytes(b"not an image")
            self.assertEqual(resolve_aspect_ratio("16:9", str(path)), "7:4")
            self.assertEqual(resolve_aspect_ratio("", str(path)), "")

    def test_manifest_pair_counts_normalizes_legacy_aspect_strings(self):
        from modules.util.dpo_curation_util import manifest_pair_counts

        manifest = {
            "pairs": [
                {"prompt": "a sunset", "aspectratio": "16:9"},
                {"prompt": "a sunset", "aspectratio": "7:4"},
            ]
        }
        counts = manifest_pair_counts(manifest)
        # Legacy "16:9" and bucket-native "7:4" are the same trainer bucket,
        # so they must count against the same group.
        self.assertEqual(counts, {("a sunset", "7:4"): 2})
