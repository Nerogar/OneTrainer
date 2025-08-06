import logging
import re
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from typing import Any

from modules.util.ui.ui_utils import (
    load_window_session_settings,
    save_window_session_settings,
    set_window_icon,
)

import customtkinter as ctk

logger = logging.getLogger(__name__)


class CaptionEditorLogic:
    """Handles the business logic for editing caption files."""

    def __init__(self, directory: str, include_subdirectories: bool = False):
        self.directory = directory
        self.include_subdirectories = include_subdirectories
        self.caption_files = self._get_caption_files()

    def _get_caption_files(self) -> list[Path]:
        """Get all caption (.txt) files in the directory."""
        base_path = Path(self.directory)
        files = list(base_path.glob("**/*.txt")) if self.include_subdirectories else list(base_path.glob("*.txt"))
        return [f for f in files if self._is_caption_file(f)]

    def _is_caption_file(self, file_path: Path) -> bool:
        """Check if this is likely a caption file (not README, etc)."""
        return any(file_path.with_suffix(ext).exists() for ext in [".jpg", ".jpeg", ".png", ".webp"])

    def _read_file_content(self, file_path: Path) -> str | None:
        """Reads a file and returns its stripped content, or None on error."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading caption file {file_path}: {e}")
            return None

    def get_preview_items(self, limit: int = 10) -> list[tuple[Path, str]]:
        """Reads a limited number of caption files for preview."""
        preview_items = []
        for file_path in self.caption_files[:limit]:
            content = self._read_file_content(file_path)
            if content is not None:
                preview_items.append((file_path, content))
        return preview_items

    def _process_and_write_file(self, file_path: Path, operation: Callable[[str], str]) -> tuple[bool, bool]:
        """
        Processes and writes a single file. Returns a tuple of (success, changed).
        """
        original = self._read_file_content(file_path)
        if original is None:
            return (False, False)

        try:
            modified = operation(original)
            if original != modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified)
                return (True, True)  # Succeeded and changed
            return (True, False)  # Succeeded, no change
        except Exception as e:
            logger.error(f"Error processing or writing caption file {file_path}: {e}")
            return (False, False)  # Failed

    def apply_changes(self, operation: Callable[[str], str]) -> tuple[int, int]:
        """Apply the given operation to all caption files."""
        count = 0
        failures = 0
        logger.info(f"Starting bulk caption edit on {len(self.caption_files)} files")

        for file_path in self.caption_files:
            success, changed = self._process_and_write_file(file_path, operation)
            if success:
                if changed:
                    count += 1
            else:
                failures += 1

        return count, failures


class BulkCaptionEditWindow(ctk.CTkToplevel):
    SESSION_SETTINGS_KEY = "bulk_caption_edit_window_settings"

    def __init__(
        self,
        parent: Any,
        directory: str,
        include_subdirectories: bool = False
    ) -> None:
        super().__init__(parent)
        self.parent = parent
        self.logic = CaptionEditorLogic(directory, include_subdirectories)

        self._setup_window()
        self._create_widgets()
        self._build_settings_map()
        self._load_session_settings()


    def _build_settings_map(self):
        """Creates a mapping from setting names to their corresponding UI variable objects."""
        self._settings_map = {
            "add_text":       self.add_text_var,
            "add_position":   self.add_position_var,
            "remove_text":    self.remove_text_var,
            "replace_text":   self.replace_text_var,
            "replace_with":   self.replace_with_var,
            "regex_pattern":  self.regex_pattern_var,
            "regex_replace":  self.regex_replace_var,
        }

    def _load_session_settings(self):
        load_window_session_settings(
            self,
            self.SESSION_SETTINGS_KEY,
            self._settings_map
        )

    def _save_session_settings(self):
        save_window_session_settings(
            self,
            self.SESSION_SETTINGS_KEY,
            self._settings_map
        )

    def destroy(self):
        self._save_session_settings()
        super().destroy()

    def _setup_window(self) -> None:

        self.title("Bulk Caption Editor")
        self.geometry("600x500")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self.transient(self.parent)
        self.attributes("-topmost", True)
        self.grab_set()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self.wait_visibility()
        self.lift()
        self.focus_force()
        self.after(200, lambda: set_window_icon(self))

    def _create_widgets(self) -> None:
        """Create all widgets for the window."""
        self._create_add_frame(0)
        self._create_remove_frame(1)
        self._create_replace_frame(2)
        self._create_regex_frame(3)
        self._create_preview_frame(4)
        self._create_button_frame(5)
        self._load_preview()

    def _create_add_frame(self, row: int) -> None:
        """Create widgets for adding text."""
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Add Text:").grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.add_text_var = ctk.StringVar()
        entry = ctk.CTkEntry(frame, textvariable=self.add_text_var)
        entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.add_position_var = ctk.StringVar(value="Prepend")
        positions = ctk.CTkOptionMenu(
            frame,
            values=["Prepend", "Append"],
            variable=self.add_position_var
        )
        positions.grid(row=0, column=2, padx=5, pady=5)

        ctk.CTkButton(frame, text="Preview", command=self._preview_add).grid(
            row=0, column=3, padx=5, pady=5
        )

    def _create_remove_frame(self, row: int) -> None:
        """Create widgets for removing text."""
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Remove Text:").grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.remove_text_var = ctk.StringVar()
        entry = ctk.CTkEntry(frame, textvariable=self.remove_text_var)
        entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkButton(frame, text="Preview", command=self._preview_remove).grid(
            row=0, column=2, padx=5, pady=5
        )

    def _create_dual_entry_frame(
        self,
        row: int,
        label1_text: str,
        var1: ctk.StringVar,
        label2_text: str,
        var2: ctk.StringVar,
        preview_command: Callable[[], None]
    ) -> None:
        """Helper to create a frame with two labeled text entries and a preview button."""
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(frame, text=label1_text).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        entry1 = ctk.CTkEntry(frame, textvariable=var1)
        entry1.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(frame, text=label2_text).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        entry2 = ctk.CTkEntry(frame, textvariable=var2)
        entry2.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        ctk.CTkButton(frame, text="Preview", command=preview_command).grid(
            row=0, column=4, padx=5, pady=5
        )

    def _create_replace_frame(self, row: int) -> None:
        """Create widgets for replacing text."""
        self.replace_text_var = ctk.StringVar()
        self.replace_with_var = ctk.StringVar()
        self._create_dual_entry_frame(
            row,
            "Replace:", self.replace_text_var,
            "With:", self.replace_with_var,
            self._preview_replace
        )

    def _create_regex_frame(self, row: int) -> None:
        """Create widgets for regex replacement."""
        self.regex_pattern_var = ctk.StringVar()
        self.regex_replace_var = ctk.StringVar()
        self._create_dual_entry_frame(
            row,
            "Regex:", self.regex_pattern_var,
            "Replace With:", self.regex_replace_var,
            self._preview_regex
        )

    def _create_preview_frame(self, row: int) -> None:
        """Create preview area."""
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=10, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Preview (first 10 captions):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )

        preview_frame = ctk.CTkScrollableFrame(frame)
        preview_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        preview_frame.grid_columnconfigure(0, weight=1)

        self.preview_frame = preview_frame

    def _create_button_frame(self, row: int) -> None:
        """Create action buttons."""
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=10, pady=10, sticky="ew")

        self.affected_count = ctk.StringVar(value="0 captions will be affected")
        ctk.CTkLabel(frame, textvariable=self.affected_count).grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )

        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(frame, text="Cancel", command=self.destroy).grid(
            row=0, column=2, padx=5, pady=5
        )

        self.apply_button = ctk.CTkButton(
            frame, text="Apply Changes", command=self._apply_changes, state="disabled"
        )
        self.apply_button.grid(row=0, column=3, padx=5, pady=5)

    def _load_preview(self) -> None:
        """Load initial preview of captions."""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

        preview_items = self.logic.get_preview_items(limit=10)

        # Now create labels with already-loaded content
        for i, (file_path, content) in enumerate(preview_items):
            label = ctk.CTkLabel(
                self.preview_frame,
                text=f"{Path(file_path).stem}: {content}",
                justify="left",
                anchor="w",
                wraplength=550
            )
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)

        self.affected_count.set(f"{len(self.logic.caption_files)} captions will be affected")

    def _preview_add(self) -> None:
        """Preview adding text to captions."""
        text = self.add_text_var.get()
        position = self.add_position_var.get()

        if not text:
            return

        self._preview_operation(
            lambda content: f"{text} {content}" if position == "Prepend" else f"{content} {text}"
        )

    def _preview_remove(self) -> None:
        """Preview removing text from captions."""
        text = self.remove_text_var.get()

        if not text:
            return

        self._preview_operation(lambda content: content.replace(text, ""))

    def _preview_replace(self) -> None:
        """Preview replacing text in captions."""
        find = self.replace_text_var.get()
        replace = self.replace_with_var.get()

        if not find:
            return

        self._preview_operation(lambda content: content.replace(find, replace))

    def _preview_regex(self) -> None:
        """Preview regex replacement in captions."""
        pattern = self.regex_pattern_var.get()
        replace = self.regex_replace_var.get()

        if not pattern:
            return

        try:
            regex = re.compile(pattern)
            self._preview_operation(lambda content: regex.sub(replace, content))
        except re.error as e:
            self._show_error(f"Invalid regex pattern: {e}")

    def _preview_operation(self, operation: Callable[[str], str]) -> None:
        """Preview an operation on the first 10 captions."""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

        changes_found = False
        preview_results = []

        preview_items = self.logic.get_preview_items(limit=10)
        for file_path, original in preview_items:
            modified = operation(original)
            if original != modified:
                changes_found = True
            preview_results.append((file_path, original, modified))

        # Now update UI with the collected results
        for i, (file_path, original, modified) in enumerate(preview_results):
            # Create label showing before/after
            label = ctk.CTkLabel(
                self.preview_frame,
                text=f"{Path(file_path).stem}:\nBefore: {original}\n\nAfter: {modified}",
                justify="left",
                anchor="w",
                wraplength=550
            )
            label.grid(row=i*2, column=0, sticky="w", padx=5, pady=(10, 0))

            if i < len(preview_results)-1:  # Don't add after the last one
                sep = ctk.CTkFrame(self.preview_frame, height=1, fg_color="gray")
                sep.grid(row=i*2+1, column=0, sticky="ew", padx=20, pady=5)

        # Enable apply button if changes were found
        if changes_found:
            self.apply_button.configure(state="normal")
            self.current_operation = operation
        else:
            self.apply_button.configure(state="disabled")
            self._show_info("No changes would be made with these settings.")

    def _apply_changes(self) -> None:
        """Apply the current operation to all caption files."""
        if not hasattr(self, 'current_operation'):
            return

        count, failures = self.logic.apply_changes(self.current_operation)

        success_msg = f"Updated {count} caption files successfully."
        if failures > 0:
            success_msg += f" ({failures} files failed)"

        logger.info(success_msg)
        self._show_info(success_msg)

        # Make sure the parent knows to refresh
        if hasattr(self.parent, 'navigation_manager'):
            try:
                # Force a refresh of the current image in the parent window
                self.parent.after(100, lambda: self.parent.navigation_manager.switch_to_image(
                    self.parent.current_image_index, from_click=False
                ))
                logger.info("Parent UI refresh scheduled")
            except Exception as e:
                logger.error(f"Error scheduling parent refresh: {e}")

        self.destroy()

    def _show_error(self, message: str) -> None:
        """Show error message."""
        tk.messagebox.showerror("Error", message, parent=self)

    def _show_info(self, message: str) -> None:
        """Show info message."""
        tk.messagebox.showinfo("Information", message, parent=self)
