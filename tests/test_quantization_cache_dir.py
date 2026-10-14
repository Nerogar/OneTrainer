import unittest

from modules.util import path_util


class TestQuantizationCacheDir(unittest.TestCase):
    def test_derived_from_cache_dir(self):
        self.assertEqual(
            path_util.quantization_cache_dir("workspace-cache/run"),
            "workspace-cache/run/quantization",
        )

    def test_trailing_slash_normalized(self):
        self.assertEqual(
            path_util.quantization_cache_dir("workspace-cache/run/"),
            "workspace-cache/run/quantization",
        )

    def test_helper_ignores_prior_baked_in_path(self):
        # Call sites must always assign from this helper (no None-guard), so a
        # previously baked-in path from another run cannot win after cache_dir changes.
        previous = "workspace-cache/A/quantization"
        rebound = path_util.quantization_cache_dir("workspace-cache/B")
        self.assertNotEqual(rebound, previous)
        self.assertEqual(rebound, "workspace-cache/B/quantization")


if __name__ == "__main__":
    unittest.main()
