import os
from tkinter import messagebox

from modules.util.dpo_curation_util import (
    apply_caption_to_pair,
    find_caption_mismatches,
    remove_finalized_pair,
)
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk
from PIL import Image


class DPOCaptionMismatchWindow(ctk.CTkToplevel):
    """Per-pair manual review of caption mismatches between chosen and rejected
    DPO sidecar .txt files. Mirrors DPOReviewWindow's image-display pattern and
    adds caption text fields plus four resolution actions: apply chosen, apply
    rejected, apply custom, or discard pair."""

    def __init__(self, parent, concept_pairs: list[tuple[str, str]], *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("Resolve Caption Mismatches")
        self.geometry("1400x900")
        self.resizable(True, True)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._concept_pairs = concept_pairs
        self._mismatches: list[dict] = []
        self._index = 0
        self._corrected = 0
        self._discarded = 0

        self._custom_text: ctk.CTkTextbox | None = None

        self.after(200, lambda: set_window_icon(self))
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self._load_mismatches()

    def _on_close(self):
        self.grab_release()
        self.destroy()

    def _load_mismatches(self):
        self._mismatches = find_caption_mismatches(self._concept_pairs)
        if not self._mismatches:
            messagebox.showinfo("No Mismatches", "No caption mismatches were found.", parent=self)
            self.destroy()
            return
        self._build_review_ui()

    def _build_review_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        if self._index >= len(self._mismatches):
            self._show_summary()
            return

        entry = self._mismatches[self._index]
        total = len(self._mismatches)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(header, text=f"Caption Mismatch  Pair {self._index + 1} / {total}", font=("", 14, "bold")).pack(
            side="left", padx=15
        )
        ctk.CTkLabel(header, text=f"Key: {entry['key']}", font=("", 12), text_color="gray").pack(side="left", padx=15)
        ctk.CTkLabel(header, text=f"Corrected: {self._corrected}", font=("", 12), text_color="green").pack(
            side="left", padx=15
        )
        ctk.CTkLabel(header, text=f"Discarded: {self._discarded}", font=("", 12), text_color="red").pack(
            side="left", padx=15
        )

        # Images side by side
        img_frame = ctk.CTkFrame(self, fg_color="transparent")
        img_frame.pack(fill="both", expand=True, padx=10, pady=5)
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

        chosen_image = entry.get("chosen_image")
        rejected_image = entry.get("rejected_image")
        if chosen_image and os.path.isfile(chosen_image):
            self._display_image(img_frame, chosen_image, row=1, col=0)
        else:
            ctk.CTkLabel(img_frame, text="(missing)").grid(row=1, column=0)
        if rejected_image and os.path.isfile(rejected_image):
            self._display_image(img_frame, rejected_image, row=1, col=1)
        else:
            ctk.CTkLabel(img_frame, text="(missing)").grid(row=1, column=1)

        # Captions side by side, each with its "Use this caption" button
        cap_frame = ctk.CTkFrame(self, fg_color="transparent")
        cap_frame.pack(fill="x", padx=10, pady=5)
        cap_frame.grid_columnconfigure(0, weight=1)
        cap_frame.grid_columnconfigure(1, weight=1)

        chosen_caption = entry.get("chosen_caption", "")
        rejected_caption = entry.get("rejected_caption", "")

        chosen_box = ctk.CTkTextbox(cap_frame, height=80, wrap="word")
        chosen_box.insert("1.0", chosen_caption)
        chosen_box.configure(state="disabled")
        chosen_box.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=2)

        rejected_box = ctk.CTkTextbox(cap_frame, height=80, wrap="word")
        rejected_box.insert("1.0", rejected_caption)
        rejected_box.configure(state="disabled")
        rejected_box.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=2)

        ctk.CTkButton(
            cap_frame,
            text="← Use this caption",
            command=lambda: self._apply_caption(chosen_caption),
        ).grid(row=1, column=0, sticky="ew", padx=(0, 5), pady=(2, 0))
        ctk.CTkButton(
            cap_frame,
            text="Use this caption →",
            command=lambda: self._apply_caption(rejected_caption),
        ).grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(2, 0))

        # Custom caption editable, pre-filled with chosen
        custom_frame = ctk.CTkFrame(self, fg_color="transparent")
        custom_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(custom_frame, text="Custom caption (applied to both sides):", font=("", 12)).pack(anchor="w")
        self._custom_text = ctk.CTkTextbox(custom_frame, height=100, wrap="word")
        self._custom_text.insert("1.0", chosen_caption)
        self._custom_text.pack(fill="x", pady=(2, 5))
        ctk.CTkButton(
            custom_frame,
            text="Apply Custom Caption",
            command=self._apply_custom,
        ).pack(fill="x")

        # Bottom action row
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(
            btn_frame,
            text="← Back",
            width=140,
            command=lambda: self._advance(-1),
            state="normal" if self._index > 0 else "disabled",
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            btn_frame,
            text="Discard Pair",
            width=160,
            fg_color="#B22222",
            command=self._discard_current,
        ).pack(side="left", padx=5, expand=True)
        ctk.CTkButton(
            btn_frame,
            text="Skip →",
            width=140,
            command=lambda: self._advance(1),
        ).pack(side="right", padx=5)

        # Keyboard
        self.bind("<Left>", lambda e: self._advance(-1) if self._index > 0 else None)
        self.bind("<Right>", lambda e: self._advance(1))
        self.bind("<Delete>", lambda e: self._discard_current())
        self.bind("<Control-Return>", lambda e: self._apply_custom())

    def _apply_caption(self, caption_text: str):
        entry = self._mismatches[self._index]
        chosen_image = entry.get("chosen_image")
        rejected_image = entry.get("rejected_image")
        try:
            apply_caption_to_pair(chosen_image, rejected_image, caption_text)
        except OSError as ex:
            messagebox.showerror("Apply Caption Error", str(ex), parent=self)
            return
        self._mismatches.pop(self._index)
        self._corrected += 1
        if self._index >= len(self._mismatches):
            self._index = max(0, len(self._mismatches) - 1)
        self._build_review_ui()

    def _apply_custom(self):
        if self._custom_text is None:
            return
        text = self._custom_text.get("1.0", "end-1c")
        self._apply_caption(text)

    def _discard_current(self):
        entry = self._mismatches[self._index]
        try:
            remove_finalized_pair(entry.get("chosen_image"), entry.get("rejected_image"))
        except OSError as ex:
            messagebox.showerror("Discard Error", str(ex), parent=self)
            return
        self._mismatches.pop(self._index)
        self._discarded += 1
        if self._index >= len(self._mismatches):
            self._index = max(0, len(self._mismatches) - 1)
        self._build_review_ui()

    def _advance(self, delta: int):
        new_index = self._index + delta
        if new_index < 0:
            new_index = 0
        self._index = new_index
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
            max_h = max(300, win_h - 500)
            with Image.open(path) as _raw:
                pil_img = self._fit_image(_raw, max_w, max_h).copy()
            ctk_img = ctk.CTkImage(light_image=pil_img, size=pil_img.size)
            label = ctk.CTkLabel(master, text="", image=ctk_img)
            label.image = ctk_img
            label.pil_image = pil_img
            label.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        except Exception:
            ctk.CTkLabel(master, text=os.path.basename(path)).grid(
                row=row,
                column=col,
                padx=10,
                pady=10,
            )

    def _show_summary(self):
        for widget in self.winfo_children():
            widget.destroy()
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(expand=True, fill="both", padx=20, pady=20)
        ctk.CTkLabel(frame, text="Caption Mismatch Review Complete", font=("", 24, "bold")).pack(pady=(0, 20))
        ctk.CTkLabel(frame, text=f"Corrected: {self._corrected}", font=("", 14), text_color="green").pack(pady=5)
        ctk.CTkLabel(frame, text=f"Discarded: {self._discarded}", font=("", 14), text_color="red").pack(pady=5)
        ctk.CTkButton(frame, text="Close", width=250, fg_color="gray", command=self.destroy).pack(pady=20)


class DPOCaptionMismatchChoiceDialog(ctk.CTkToplevel):
    """Three-way prompt: Correct All to Chosen, Manually Review, or Cancel.
    Returns the chosen action via `result` ('correct_all' | 'manual' | None)."""

    def __init__(self, parent, mismatch_count: int):
        super().__init__(parent)
        self.title("Caption Mismatches Found")
        self.geometry("520x220")
        self.resizable(False, False)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.result: str | None = None

        ctk.CTkLabel(
            self,
            text=f"Found {mismatch_count} pair(s) where the chosen and rejected\n"
            f"caption files differ.\n\n"
            f"How would you like to resolve them?",
            font=("", 13),
            justify="center",
        ).pack(padx=20, pady=(20, 15))

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkButton(
            btn_frame,
            text="Correct All to Chosen",
            width=150,
            command=lambda: self._set_result("correct_all"),
        ).pack(side="left", padx=5, expand=True, fill="x")
        ctk.CTkButton(
            btn_frame,
            text="Manually Review",
            width=150,
            command=lambda: self._set_result("manual"),
        ).pack(side="left", padx=5, expand=True, fill="x")
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            width=150,
            fg_color="gray",
            command=self._cancel,
        ).pack(side="left", padx=5, expand=True, fill="x")

        self.after(200, lambda: set_window_icon(self))
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

    def _set_result(self, value: str):
        self.result = value
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.grab_release()
        self.destroy()
