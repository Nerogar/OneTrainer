from unittest.mock import MagicMock

from modules.ui.BulkCaptionEditWindow import CaptionEditorLogic

import pytest

from .helpers import setup_test_directory


class TestCaptionEditorLogic:
    def test_init_finds_caption_files(self, tmp_path):
        """Tests that only valid caption files in the root are found."""
        file_specs = [
            ("img_a.png", True),
            ("img_a.txt", False, {'content': "caption one"}),
            ("img_b.jpg", True),
            ("img_b.txt", False, {'content': "caption two"}),
            ("subdir/img_c.txt", False, {'content': "in subdir"}),
        ]
        d, _ = setup_test_directory(tmp_path, "logic_root_test", file_specs)

        logic = CaptionEditorLogic(str(d), include_subdirectories=False)
        assert len(logic.caption_files) == 2
        filenames = {p.name for p in logic.caption_files}
        assert "img_a.txt" in filenames
        assert "img_b.txt" in filenames

    def test_init_with_subdirectories(self, tmp_path):
        """Tests that subdirectories are included when specified."""
        file_specs = [
            ("img_a.png", True),
            ("img_a.txt", False, {'content': "caption one"}),
            ("subdir/img_c.webp", True),
            ("subdir/img_c.txt", False, {'content': "caption three in subdir"}),
        ]
        d, _ = setup_test_directory(tmp_path, "logic_subdir_test", file_specs)

        logic = CaptionEditorLogic(str(d), include_subdirectories=True)
        assert len(logic.caption_files) == 2
        filenames = {p.name for p in logic.caption_files}
        assert "img_a.txt" in filenames
        assert "img_c.txt" in filenames

    def test_get_preview_items(self, tmp_path):
        """Tests that preview items are correctly read."""
        file_specs = [
            ("1.png", True), ("1.txt", False, {'content': "caption one"}),
            ("2.png", True), ("2.txt", False, {'content': "caption two"}),
        ]
        d, _ = setup_test_directory(tmp_path, "logic_preview_test", file_specs)
        logic = CaptionEditorLogic(str(d))
        preview = logic.get_preview_items(limit=2)
        assert len(preview) == 2
        assert preview[0][1] == "caption one"
        assert preview[1][1] == "caption two"

    @pytest.mark.parametrize(
        "operation, expected_content_a, expected_content_b",
        [
            (lambda c: f"new_tag, {c}", "new_tag, caption one", "new_tag, caption two, with a tag"),
            (lambda c: c.replace("caption", "text"), "text one", "text two, with a tag"),
            (lambda c: c.replace(" non_existent", ""), "caption one", "caption two, with a tag"),
        ],
    )
    def test_apply_changes(self, tmp_path, operation, expected_content_a, expected_content_b):
        """Tests applying various operations to caption files."""
        file_specs = [
            ("img_a.png", True), ("img_a.txt", False, {'content': "caption one"}),
            ("img_b.jpg", True), ("img_b.txt", False, {'content': "caption two, with a tag"}),
        ]
        d, _ = setup_test_directory(tmp_path, "logic_apply_test", file_specs)
        logic = CaptionEditorLogic(str(d), include_subdirectories=False)

        original_a = (d / "img_a.txt").read_text(encoding='utf-8')
        original_b = (d / "img_b.txt").read_text(encoding='utf-8')
        expected_changes = 0
        if original_a != expected_content_a:
            expected_changes += 1
        if original_b != expected_content_b:
            expected_changes += 1

        count, failures = logic.apply_changes(operation)

        assert failures == 0
        assert count == expected_changes
        assert (d / "img_a.txt").read_text(encoding='utf-8') == expected_content_a
        assert (d / "img_b.txt").read_text(encoding='utf-8') == expected_content_b

    def test_error_handling_on_read(self, tmp_path, monkeypatch):
        """Tests that read errors are handled gracefully."""
        file_specs = [("1.png", True), ("1.txt", False)]
        d, _ = setup_test_directory(tmp_path, "logic_error_test", file_specs)
        logic = CaptionEditorLogic(str(d))

        mock_read = MagicMock(return_value=None)
        monkeypatch.setattr(logic, "_read_file_content", mock_read)

        count, failures = logic.apply_changes(lambda c: c)
        assert count == 0
        assert failures == len(logic.caption_files)

    def test_apply_changes_with_utf8_characters(self, tmp_path):
            """Tests that UTF-8 characters (JP, CN, KR) are handled correctly."""
            file_specs = [
                ("img_jp.png", True), ("img_jp.txt", False, {'content': "これは日本語のキャプションです"}),
                ("img_cn.png", True), ("img_cn.txt", False, {'content': "这是一个中文标题"}),
                ("img_kr.png", True), ("img_kr.txt", False, {'content': "이것은 한국어 캡션입니다"}),
            ]
            d, _ = setup_test_directory(tmp_path, "logic_utf8_test", file_specs)
            logic = CaptionEditorLogic(str(d))

            def operation(c):
                return f"new_tag, {c}"

            count, failures = logic.apply_changes(operation)

            assert failures == 0
            assert count == 3
            assert (d / "img_jp.txt").read_text(encoding='utf-8') == "new_tag, これは日本語のキャプションです"
            assert (d / "img_cn.txt").read_text(encoding='utf-8') == "new_tag, 这是一个中文标题"
            assert (d / "img_kr.txt").read_text(encoding='utf-8') == "new_tag, 이것은 한국어 캡션입니다"

    def test_handling_of_files_with_invalid_encoding(self, tmp_path):
        """Tests that files with invalid byte sequences are handled as failures."""
        d = tmp_path / "logic_invalid_encoding_test"
        d.mkdir()

        # Create a valid file to ensure some operations succeed
        (d / "valid.txt").write_text("valid content", encoding='utf-8')
        (d / "valid.png").touch()

        # Create a file with an invalid UTF-8 byte sequence
        invalid_file_path = d / "invalid.txt"
        with open(invalid_file_path, "wb") as f:
            f.write(b"start " + b"\xff" + b" end")
        (d / "invalid.png").touch()

        logic = CaptionEditorLogic(str(d))

        # The logic should identify both .txt files
        assert len(logic.caption_files) == 2

        count, failures = logic.apply_changes(lambda c: f"prefix {c}")

        # The valid file should be changed, the invalid one should fail to be read.
        assert count == 1
        assert failures == 1
        assert (d / "valid.txt").read_text(encoding='utf-8') == "prefix valid content"
