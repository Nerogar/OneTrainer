import os
from tkinter import messagebox

from modules.util.dpo_curation_util import (
    remove_finalized_pair,
    scan_finalized_pairs,
)
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk
from PIL import Image


class DPOReviewWindow(ctk.CTkToplevel):
    def __init__(self, parent, concept_pairs: list[tuple[str, str]], *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("Review DPO Pairs")
        self.geometry("1400x900")
        self.resizable(True, True)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._concept_pairs = concept_pairs
        self._pairs: list[dict] = []
        self._index = 0
        self._removed = 0

        self.after(200, lambda: set_window_icon(self))
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self._load_pairs()

    def _on_close(self):
        self.grab_release()
        self.destroy()

    def _safe_grab(self):
        try:
            self.grab_set()
            self.focus_set()
        except Exception:
            pass

    def _load_pairs(self):
        self._pairs = scan_finalized_pairs(self._concept_pairs)
        if not self._pairs:
            messagebox.showinfo("No Pairs", "No image pairs found in the configured concept folders.")
            self.destroy()
            return

        orphans = [p for p in self._pairs if p["is_orphan"]]
        if orphans:
            result = messagebox.askyesno(
                "Orphaned Pairs Found",
                f"Found {len(orphans)} pair(s) with a missing chosen or rejected image.\n\nRemove them now?",
                parent=self,
            )
            if result:
                for entry in orphans:
                    remove_finalized_pair(entry.get("chosen_path"), entry.get("rejected_path"))
                self._pairs = [p for p in self._pairs if not p["is_orphan"]]
                self._removed += len(orphans)
                if not self._pairs:
                    self._show_summary()
                    return

        self._build_review_ui()

    def _build_review_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        if self._index >= len(self._pairs):
            self._show_summary()
            return

        entry = self._pairs[self._index]
        total = len(self._pairs)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(header, text=f"Pair {self._index + 1} / {total}", font=("", 14, "bold")).pack(side="left", padx=15)
        ctk.CTkLabel(header, text=f"Key: {entry['key']}", font=("", 12), text_color="gray").pack(side="left", padx=15)
        ctk.CTkLabel(header, text=f"Removed: {self._removed}", font=("", 12), text_color="red").pack(
            side="left", padx=15
        )

        # Images side by side
        img_frame = ctk.CTkFrame(self, fg_color="transparent")
        img_frame.pack(expand=True, fill="both", padx=10, pady=5)
        img_frame.grid_columnconfigure(0, weight=1)
        img_frame.grid_columnconfigure(1, weight=1)
        img_frame.grid_rowconfigure(0, weight=0)
        img_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(img_frame, text="Chosen", font=("", 14, "bold"), text_color="green").grid(
            row=0, column=0, pady=(0, 5)
        )
        ctk.CTkLabel(img_frame, text="Rejected", font=("", 14, "bold"), text_color="red").grid(
            row=0, column=1, pady=(0, 5)
        )

        chosen_path = entry.get("chosen_path")
        rejected_path = entry.get("rejected_path")

        if chosen_path and os.path.isfile(chosen_path):
            self._display_image(img_frame, chosen_path, row=1, col=0)
        else:
            ctk.CTkLabel(img_frame, text="(missing)", font=("", 14), text_color="gray").grid(row=1, column=0)
        if rejected_path and os.path.isfile(rejected_path):
            self._display_image(img_frame, rejected_path, row=1, col=1)
        else:
            ctk.CTkLabel(img_frame, text="(missing)", font=("", 14), text_color="gray").grid(row=1, column=1)

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(
            btn_frame,
            text="\u2190 Back",
            width=150,
            command=lambda: self._advance(-1),
            state="normal" if self._index > 0 else "disabled",
        ).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Remove", width=150, fg_color="#B22222", command=self._remove_current).pack(
            side="left", padx=10, expand=True
        )
        ctk.CTkButton(btn_frame, text="Keep \u2192", width=150, command=lambda: self._advance(1)).pack(
            side="right", padx=10
        )

        # Keyboard bindings
        self.bind("<Left>", lambda e: self._advance(-1) if self._index > 0 else None)
        self.bind("<Right>", lambda e: self._advance(1))
        self.bind("<Delete>", lambda e: self._remove_current())

    def _advance(self, delta: int):
        self._index += delta
        if self._index < 0:
            self._index = 0
        self._build_review_ui()

    def _remove_current(self):
        entry = self._pairs[self._index]
        remove_finalized_pair(entry.get("chosen_path"), entry.get("rejected_path"))
        self._pairs.pop(self._index)
        self._removed += 1
        if self._index >= len(self._pairs):
            self._index = max(0, len(self._pairs) - 1)
        self._build_review_ui()

    def _fit_image(self, pil_img: Image.Image, max_w: int, max_h: int) -> Image.Image:
        scale = min(max_w / pil_img.width, max_h / pil_img.height)
        new_w = max(1, int(pil_img.width * scale))
        new_h = max(1, int(pil_img.height * scale))
        return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _display_image(self, master, path: str, row: int, col: int):
        try:
            self.update_idletasks()
            win_w = self.winfo_width() or self.winfo_screenwidth()
            win_h = self.winfo_height() or self.winfo_screenheight()
            max_w = max(400, win_w // 2 - 40)
            max_h = max(400, win_h - 200)
            with Image.open(path) as _raw:
                pil_img = self._fit_image(_raw, max_w, max_h).copy()
            ctk_img = ctk.CTkImage(light_image=pil_img, size=pil_img.size)

            label = ctk.CTkLabel(master, text="", image=ctk_img)
            label.image = ctk_img
            label.pil_image = pil_img
            label.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        except Exception:
            ctk.CTkLabel(master, text=os.path.basename(path)).grid(row=row, column=col, padx=10, pady=10)

    def _show_summary(self):
        for widget in self.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(frame, text="Review Complete", font=("", 24, "bold")).pack(pady=(0, 20))

        remaining = len(self._pairs)
        ctk.CTkLabel(frame, text=f"Kept: {remaining} pairs", font=("", 14)).pack(pady=5)
        ctk.CTkLabel(frame, text=f"Removed: {self._removed} pairs", font=("", 14), text_color="red").pack(pady=5)

        ctk.CTkButton(frame, text="Close", width=250, fg_color="gray", command=self.destroy).pack(pady=20)
