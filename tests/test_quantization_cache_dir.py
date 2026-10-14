import unittest

from modules.util import path_util


class TestQuantizationCacheDir(unittest.TestCase):
    def test_derived_from_cache_dir(self):
        self.assertEqual(
            path_util.quantization_cache_dir("workspace-cache/run"),
            "workspace-cache/run/quantization",
        )

    def test_stale_value_overwritten_by_rebind(self):
        # Reproduce #1707: a baked-in path from a prior run must not win after
        # cache_dir changes (Save As / duplicated config).
        stale = "workspace-cache/A/quantization"
        new_cache_dir = "workspace-cache/B"

        class Quantization:
            cache_dir = stale

        class Config:
            cache_dir = new_cache_dir
            quantization = Quantization()

        config = Config()
        # Fixed call-site semantics (always rebind, no None-guard):
        config.quantization.cache_dir = path_util.quantization_cache_dir(config.cache_dir)

        self.assertEqual(config.quantization.cache_dir, "workspace-cache/B/quantization")
        self.assertNotEqual(config.quantization.cache_dir, stale)


if __name__ == "__main__":
    unittest.main()
