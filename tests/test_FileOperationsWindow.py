from queue import Queue
from unittest.mock import MagicMock

from modules.ui.FileOperationsWindow import (
    ONE_MEGAPIXEL,
    FileOperationsWindow,
    FileProcessor,
)

import pytest
from PIL import Image

from .helpers import (
    assert_message_in_queue,
    create_dummy_image,
    create_dummy_text_file,
    setup_test_directory,
)

# Helpers

def process_files_in_directory(processor, directory, method_name):
    """Collect files from directory and process them with the specified method.

    Args:
        processor: FileProcessor instance
        directory: Path to directory containing files
        method_name: String name of the processor method to call

    Returns:
        The result of the processor method
    """
    all_files = list(directory.glob("*"))
    method = getattr(processor, method_name)
    return method(all_files)

# Fixtures

@pytest.fixture
def base_config():
    return {
        "resize_megapixels": "Compute Proof (4MP)",
        "custom_megapixels": 1.0,
        "process_alpha": False,
        "alpha_bg_color": "#FFFFFF",
    }


@pytest.fixture
def file_processor(base_config):
    """Returns a FileProcessor instance with a default config and a single worker."""
    message_queue = Queue()
    def cancel_requested():
        return False
    return FileProcessor(
        base_config, message_queue, cancel_requested, max_workers=1
    )

# Tests

def test_calculate_dimensions_for_megapixels():
    """Tests the static method for calculating new dimensions based on target megapixels."""
    w, h = FileProcessor.calculate_dimensions_for_megapixels(
        2000, 1000, ONE_MEGAPIXEL
    )
    assert w == 1448 and h == 724

    w, h = FileProcessor.calculate_dimensions_for_megapixels(
        1000, 2000, ONE_MEGAPIXEL
    )
    assert w == 724 and h == 1448

    w, h = FileProcessor.calculate_dimensions_for_megapixels(
        2000, 2000, ONE_MEGAPIXEL
    )
    assert w == 1024 and h == 1024


class TestFileProcessor:
    def test_rename_files_sequentially(self, file_processor, tmp_path):
        file_specs = [
            ("b.png", True),
            ("b.txt", False, {'content': "caption for b"}),
            ("a.jpg", True),
            ("a.txt", False, {'content': "caption for a"}),
            ("a-masklabel.png", True),
            ("c.webp", True)
        ]

        d, _ = setup_test_directory(tmp_path, "rename_test", file_specs)
        renamed_files = process_files_in_directory(file_processor, d, "rename_files_sequentially")

        # Rest of the assertions remain the same
        expected_renamed_stems = {"1", "2", "3"}
        final_files = {p.name for p in d.glob("*")}

        assert {p.stem for p in renamed_files} == expected_renamed_stems
        assert final_files == {
            "1.jpg", "1.txt", "1-masklabel.png",
            "2.png", "2.txt", "3.webp"
        }

    def test_rename_files_already_sequential(self, file_processor, tmp_path):
        """Tests that renaming is skipped if files are already in sequential order."""
        file_specs = [
            ("1.png", True),
            ("1.txt", False),
            ("2.jpg", True)
        ]
        d, all_files = setup_test_directory(tmp_path, "seq_test", file_specs)

        file_processor.rename_files_sequentially(all_files)

        assert assert_message_in_queue(
            file_processor.message_queue,
            "log",
            "Files are already named sequentially. No action needed."
        )

    def test_resize_large_images(self, file_processor, tmp_path):
        """Tests resizing of images that exceed the megapixel threshold."""
        # Create a directory with a large image and a small image
        file_specs = [
            ("large_image.png", True, {"width": 3000, "height": 3000}),  # 9MP image
            ("small_image.jpg", True)  # Default 10x10 image
        ]
        d, created_files = setup_test_directory(tmp_path, "resize_test", file_specs)

        large_image_path = d / "large_image.png"
        small_image_path = d / "small_image.jpg"

        original_large_size = Image.open(large_image_path).size
        assert original_large_size == (3000, 3000)

        process_files_in_directory(file_processor, d, "resize_large_images")

        resized_image = Image.open(large_image_path)
        # For 4MP threshold (4,194,304 pixels), a square image becomes 2048x2048
        assert resized_image.size == (2048, 2048)

        # Verify smaller images were not touched
        assert Image.open(small_image_path).size == (10, 10)

    def test_convert_to_webp(self, file_processor, tmp_path):
        """Tests image conversion to WebP format."""
        file_specs = [
            ("test.png", True),
            ("another.jpg", True, {"format": "JPEG"})
        ]
        d, created_files = setup_test_directory(tmp_path, "convert_test", file_specs)

        png_path = d / "test.png"
        jpg_path = d / "another.jpg"

        process_files_in_directory(file_processor, d, "convert_to_webp")

        assert not png_path.exists()
        assert not jpg_path.exists()
        assert (d / "test.webp").exists()
        assert (d / "another.webp").exists()

    def test_verify_images(self, file_processor, tmp_path):
        """Tests the image verification process for valid and corrupt files."""
        file_specs = [
            ("valid.png", True),
            ("corrupt.png", False)  # Text file with image extension
        ]
        d, created_files = setup_test_directory(tmp_path, "verify_test", file_specs)

        results = file_processor.verify_images(created_files)

        assert isinstance(results, dict)
        assert results.get("valid") == 1
        assert results.get("errors") == 1

    def test_process_alpha_images_replace_color(self, file_processor, tmp_path):
        """Tests replacing transparency with a solid background color."""
        # Creating a transparent image with a red square in the middle
        file_specs = [
            ("transparent.png", True, {"mode": "RGBA", "width": 10, "height": 10})
        ]
        d, created_files = setup_test_directory(tmp_path, "alpha_test", file_specs)
        alpha_image_path = created_files[0]

        # Create transparent background with red square in middle
        img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        for x in range(2, 8):
            for y in range(2, 8):
                img.putpixel((x, y), (255, 0, 0, 255))
        img.save(alpha_image_path)

        bg_color_hex = "#0000FF"
        bg_color_rgb = (0, 0, 255)
        file_processor.config_data["alpha_bg_color"] = bg_color_hex

        file_processor.process_alpha_images([alpha_image_path])

        with Image.open(alpha_image_path) as processed_img:
            # 1. The mode should now be RGB (alpha channel removed)
            assert processed_img.mode == "RGB"

            # 2. The transparent corners should now be the new background color
            assert processed_img.getpixel((0, 0)) == bg_color_rgb
            assert processed_img.getpixel((9, 9)) == bg_color_rgb

            assert processed_img.getpixel((5, 5)) == (255, 0, 0)

    def test_verify_single_image(self, tmp_path):
        """Standalone test for verify_single_image using pytest.raises."""
        q = Queue()
        fp = FileProcessor({}, q, lambda: False, max_workers=1)

        # Valid image
        valid_img_path = create_dummy_image(tmp_path / "valid.png")
        try:
            fp.verify_single_image(valid_img_path)
        except ValueError:
            pytest.fail("verify_single_image raised ValueError unexpectedly for a valid image.")

        # Corrupt image should raise ValueError
        corrupt_img_path = create_dummy_text_file(tmp_path / "corrupt.png")
        with pytest.raises(ValueError, match="is corrupt or invalid"):
            fp.verify_single_image(corrupt_img_path)

    def test_optimize_pngs(self, file_processor, tmp_path):
        """Standalone test for optimize_pngs."""
        file_specs = [
            ("a.png", True, {"width": 20, "height": 20}),
            ("b.png", True, {"width": 20, "height": 20})
        ]
        d, created_files = setup_test_directory(tmp_path, "opt_test", file_specs)

        file_processor.optimize_pngs(created_files)

        # Check for expected log messages
        assert assert_message_in_queue(
            file_processor.message_queue,
            "log",
            "Starting PNG optimization"
        )
        assert assert_message_in_queue(
            file_processor.message_queue,
            "log",
            "Completed optimization:"
        )

class TestFileOperationsWindowUI:
    """Tests for the FileOperationsWindow class, mocking UI interactions."""

    @pytest.mark.parametrize(
        "dir_path, expected_log",
        [
            ("", "Error: No directory selected"),
            ("non_existent_dir", "is not a valid directory"),
            ("file_path", "is not a valid directory"),
        ],
    )
    def test_process_files_with_invalid_directory(
        self, tmp_path, dir_path, expected_log
    ):
        """Test that _process_files rejects invalid directories."""
        mock_window = MagicMock(spec=FileOperationsWindow)
        mock_window.message_queue = Queue()
        mock_window.config_data = {}  # Needs to exist for the check

        # Set up the specific path for the test case
        if dir_path == "file_path":
            file = tmp_path / "file.txt"
            file.touch()
            path_to_test = str(file)
        elif dir_path == "non_existent_dir":
            path_to_test = str(tmp_path / "non_existent_dir")
        else:
            path_to_test = ""

        mock_window.dir_path_var = MagicMock(get=lambda: path_to_test)

        mock_window._log = MagicMock(
            side_effect=lambda msg: mock_window.message_queue.put(("log", msg))
        )

        FileOperationsWindow._process_files(mock_window)

        assert assert_message_in_queue(
            mock_window.message_queue, "log", expected_log
        )

class TestFileProcessorEdgeCases:
    """Tests for edge cases and graceful failure in the FileProcessor."""

    @pytest.mark.parametrize(
        "invalid_color", ["not-a-color", "#GGHHII", "#12345", "123"]
    )
    def test_process_alpha_with_invalid_color(self, tmp_path, invalid_color):
        """Tests that invalid hex/color values are handled gracefully."""
        d = tmp_path / "alpha_invalid"
        d.mkdir()
        alpha_image_path = d / "test.png"
        img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        # Add an opaque blue sqaure to the image
        for x in range(2, 8):
            for y in range(2, 8):
                img.putpixel((x, y), (0, 0, 255, 255))  # Opaque blue
        img.save(alpha_image_path)

        message_queue = Queue()
        config = {"alpha_bg_color": invalid_color}
        processor = FileProcessor(config, message_queue, lambda: False, 1)

        processor.process_alpha_images([alpha_image_path])

        assert assert_message_in_queue(
            message_queue, "log", f"Invalid color '{invalid_color}'"
        )

        # Assert that the image was processed with a white background as a fallback
        with Image.open(alpha_image_path) as processed_img:
            assert processed_img.mode == "RGB"
            assert processed_img.getpixel((0, 0)) == (255, 255, 255)
            assert processed_img.getpixel((5, 5)) == (0, 0, 255)
