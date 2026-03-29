import math
import os
import random
from collections import defaultdict
from tkinter import filedialog, messagebox

from modules.util import path_util
from modules.util.dpo_curation_util import export_curated_pairs, has_existing_exports
from modules.util.image_metadata_util import extract_metadata
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk
from PIL import Image


class DPOCurationWindow(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("DPO Pair Tool")
        self.geometry("1400x900")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.groups: list[dict] = []
        self.current_group_index = 0
        self.results: dict[int, list[dict]] = {}  # group_index -> [{chosen, rejected}, ...]
        self.current_remaining_images: list[str] = []
        self.pairs_per_group = 1
        self.pairs_created_in_group = 0

        # ELO state
        self.elo_ratings: dict[str, float] = {}
        self.elo_comparisons_done = 0
        self.elo_pair: tuple[str, str] | None = None

        # Selection state
        self.selection_phase = "best"  # "best" or "worst"
        self.selected_best: str | None = None

        self.mode = "selection"  # "elo" or "selection"
        self.pairs_per_group_var = ctk.StringVar(value="1")

        self._build_start_ui()

        self.after(200, lambda: set_window_icon(self))
        self.focus_set()

    def _on_close(self):
        self.destroy()

    def _build_start_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(frame, text="DPO Pair Tool", font=("", 24, "bold")).pack(pady=(0, 20))
        ctk.CTkLabel(frame, text="Select a folder of generated images.\n"
                     "Prompt and aspect ratio will be extracted from image metadata (SwarmUI format).\n"
                     "Images are grouped by (prompt, aspect ratio) for comparison.\n"
                     "The tool exports chosen/rejected folders for DPO training.",
                     justify="center").pack(pady=(0, 20))

        pair_count_frame = ctk.CTkFrame(frame, fg_color="transparent")
        pair_count_frame.pack(pady=(0, 20))
        ctk.CTkLabel(pair_count_frame, text="Pairs per group:").pack(side="left", padx=(0, 10))
        ctk.CTkEntry(pair_count_frame, textvariable=self.pairs_per_group_var, width=60).pack(side="left")
        ctk.CTkLabel(pair_count_frame, text="How many pairs to collect before moving on.", anchor="w").pack(side="left", padx=(10, 0))

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(pady=10)

        ctk.CTkButton(btn_frame, text="Select Folder & Start (ELO)", width=250,
                      command=lambda: self._select_folder("elo")).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Select Folder & Start (Selection)", width=250,
                      command=lambda: self._select_folder("selection")).pack(side="left", padx=10)

    def _select_folder(self, mode: str):
        folder = filedialog.askdirectory(title="Select folder of generated images")
        if not folder:
            return

        self.mode = mode
        self._scan_and_group(folder)

        if not self.groups:
            messagebox.showwarning("No Groups", "No images with extractable prompt metadata found.")
            return

        self.pairs_per_group = self._parse_pairs_per_group()
        self.current_group_index = 0
        self.results = {}

        if mode == "elo":
            self._start_group_round()
        else:
            self._start_group_round()

    def _scan_and_group(self, folder: str):
        groups_dict: dict[tuple[str, str], list[str]] = defaultdict(list)

        for root, _, files in os.walk(folder):
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in path_util.supported_image_extensions():
                    continue
                path = os.path.join(root, filename)
                meta = extract_metadata(path)
                prompt = meta.get('prompt', '').strip()
                ar = meta.get('aspectratio', '').strip()
                if prompt:
                    groups_dict[(prompt, ar)].append(path)

        self.groups = []
        for (prompt, ar), images in sorted(groups_dict.items()):
            if len(images) >= 2:
                self.groups.append({
                    'prompt': prompt,
                    'aspectratio': ar,
                    'images': images,
                })

    def _parse_pairs_per_group(self) -> int:
        try:
            value = int(str(self.pairs_per_group_var.get()).strip())
        except (TypeError, ValueError):
            return 1
        return max(1, value)

    def _start_group_round(self):
        if self.current_group_index >= len(self.groups):
            self._show_export()
            return

        group = self.groups[self.current_group_index]
        self.current_remaining_images = list(group['images'])
        self.pairs_created_in_group = len(self.results.get(self.current_group_index, []))
        if len(self.current_remaining_images) < 2 or self.pairs_created_in_group >= self.pairs_per_group:
            self._advance_group()
            return

        if self.mode == "elo":
            self._start_elo_round()
        else:
            self._start_selection_round()

    # ---- ELO Mode ----

    def _start_elo_round(self):
        self.elo_ratings = dict.fromkeys(self.current_remaining_images, 1500.0)
        self.elo_comparisons_done = 0
        self._elo_next_pair()

    def _elo_suggested_comparisons(self) -> int:
        n = len(self.current_remaining_images)
        return max(15, math.ceil(n * math.log2(max(n, 2))))

    def _elo_next_pair(self):
        images = self.current_remaining_images

        sorted_imgs = sorted(images, key=lambda x: self.elo_ratings[x])
        if len(sorted_imgs) < 2:
            self._advance_group()
            return

        idx = random.randint(0, len(sorted_imgs) - 2)
        self.elo_pair = (sorted_imgs[idx], sorted_imgs[idx + 1])
        self._build_elo_ui()

    def _build_elo_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        group = self.groups[self.current_group_index]
        suggested = self._elo_suggested_comparisons()

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(header, text=f"Prompt: {group['prompt'][:80]}...",
                     font=("", 12), anchor="w").pack(side="left", padx=5)
        ctk.CTkLabel(header, text=f"AR: {group['aspectratio']}",
                     font=("", 12)).pack(side="left", padx=15)
        ctk.CTkLabel(header, text=f"Group {self.current_group_index + 1}/{len(self.groups)}",
                     font=("", 12)).pack(side="left", padx=15)
        ctk.CTkLabel(header, text=f"Comparisons: {self.elo_comparisons_done}/{suggested} suggested",
                     font=("", 12)).pack(side="left", padx=15)

        # Images
        img_frame = ctk.CTkFrame(self, fg_color="transparent")
        img_frame.pack(expand=True, fill="both", padx=10, pady=5)
        img_frame.grid_columnconfigure(0, weight=1)
        img_frame.grid_columnconfigure(1, weight=1)
        img_frame.grid_rowconfigure(0, weight=1)

        for col, path in enumerate(self.elo_pair):
            self._display_image(img_frame, path, row=0, col=col)
            rating = self.elo_ratings.get(path, 1500.0)
            ctk.CTkLabel(img_frame, text=f"ELO: {rating:.0f}",
                         font=("", 12)).grid(row=1, column=col, pady=(0, 5))

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(btn_frame, text="A is Better (←)", width=200,
                      command=lambda: self._elo_vote("a")).pack(side="left", padx=20, expand=True)
        ctk.CTkButton(btn_frame, text="Tie / Skip (↓)", width=200,
                      command=lambda: self._elo_vote("tie")).pack(side="left", padx=20, expand=True)
        ctk.CTkButton(btn_frame, text="B is Better (→)", width=200,
                      command=lambda: self._elo_vote("b")).pack(side="left", padx=20, expand=True)
        ctk.CTkButton(btn_frame, text="Skip Group", width=150, fg_color="#8B4513",
                      command=self._skip_group).pack(side="right", padx=10)
        ctk.CTkButton(btn_frame, text="Accept Pair", width=150, fg_color="gray",
                      command=self._elo_finish_round).pack(side="right", padx=10)

        # Keyboard bindings
        self.bind("<Left>", lambda e: self._elo_vote("a"))
        self.bind("<Right>", lambda e: self._elo_vote("b"))
        self.bind("<Down>", lambda e: self._elo_vote("tie"))

    def _elo_vote(self, winner: str):
        a, b = self.elo_pair
        K = 32.0

        if winner == "a":
            sa, sb = 1.0, 0.0
        elif winner == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        ea = 1.0 / (1.0 + 10.0 ** ((self.elo_ratings[b] - self.elo_ratings[a]) / 400.0))
        eb = 1.0 - ea
        self.elo_ratings[a] += K * (sa - ea)
        self.elo_ratings[b] += K * (sb - eb)

        self.elo_comparisons_done += 1
        self._elo_next_pair()

    def _elo_finish_round(self):
        sorted_imgs = sorted(self.elo_ratings.keys(), key=lambda x: self.elo_ratings[x], reverse=True)
        best = sorted_imgs[0]
        worst = sorted_imgs[-1]
        best_rating = self.elo_ratings[best]
        worst_rating = self.elo_ratings[worst]
        best_name = os.path.basename(best)
        worst_name = os.path.basename(worst)
        if not messagebox.askyesno("Accept Pair",
                                    f"Best: {best_name} (ELO {best_rating:.0f})\n"
                                    f"Worst: {worst_name} (ELO {worst_rating:.0f})\n\n"
                                    f"Accept this pair?"):
            return
        self._register_pair(best, worst)

    # ---- Selection Mode ----

    def _start_selection_round(self):
        self.selection_phase = "best"
        self.selected_best = None
        self._build_selection_ui()

    def _build_selection_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        group = self.groups[self.current_group_index]

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(header, text=f"Prompt: {group['prompt'][:80]}...",
                     font=("", 12), anchor="w").pack(side="left", padx=5)
        ctk.CTkLabel(header, text=f"AR: {group['aspectratio']}",
                     font=("", 12)).pack(side="left", padx=15)
        ctk.CTkLabel(header, text=f"Group {self.current_group_index + 1}/{len(self.groups)}",
                     font=("", 12)).pack(side="left", padx=15)

        ctk.CTkButton(header, text="Skip Group", width=120, fg_color="#8B4513",
                      command=self._skip_group).pack(side="right", padx=10)

        if self.selection_phase == "best":
            phase_text = (f"Pair {self.pairs_created_in_group + 1}/{self.pairs_per_group}. "
                          "Click to view, then right-click to select as BEST")
        else:
            best_name = os.path.basename(self.selected_best) if self.selected_best else "?"
            phase_text = (f"Pair {self.pairs_created_in_group + 1}/{self.pairs_per_group}. "
                          f"Best: {best_name}. Now right-click to select WORST")
        ctk.CTkLabel(header, text=phase_text, font=("", 14, "bold"),
                     text_color="green" if self.selection_phase == "best" else "red").pack(side="right", padx=15)

        # Thumbnail grid
        grid_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        grid_frame.pack(expand=True, fill="both", padx=10, pady=5)

        images = self.current_remaining_images
        cols = max(1, min(5, int(self.winfo_width() / 300) or 4))
        visible = 0
        for path in images:
            if path == self.selected_best:
                continue
            row = visible // cols
            col = visible % cols
            visible += 1
            grid_frame.grid_columnconfigure(col, weight=1)
            self._display_thumbnail(grid_frame, path, row, col)

    def _display_thumbnail(self, master, path: str, row: int, col: int):
        thumb_size = 250
        try:
            pil_img = Image.open(path)
            pil_img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, size=pil_img.size)

            label = ctk.CTkLabel(master, text="", image=ctk_img)
            label.image = ctk_img  # prevent GC
            label.pil_image = pil_img  # prevent GC - CTkImage needs the PIL data alive for DPI re-rendering
            label.grid(row=row, column=col, padx=5, pady=5)
            label.bind("<Button-1>", lambda e, p=path: self._selection_preview(p))
            label.bind("<Button-3>", lambda e, p=path: self._selection_pick(p))
        except Exception:
            ctk.CTkLabel(master, text=os.path.basename(path)).grid(row=row, column=col, padx=5, pady=5)

    def _selection_preview(self, path: str):
        preview = ctk.CTkToplevel(self)
        preview.title("Preview — Right-click to select")
        preview.attributes("-fullscreen", True)
        preview.focus_set()

        try:
            pil_img = Image.open(path)
            sw, sh = preview.winfo_screenwidth(), preview.winfo_screenheight()
            pil_img.thumbnail((sw, sh - 50), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, size=pil_img.size)

            label = ctk.CTkLabel(preview, text="", image=ctk_img)
            label.image = ctk_img  # prevent GC
            label.pil_image = pil_img  # prevent GC
            label.pack(expand=True)

            def on_select(e):
                preview.destroy()
                self._selection_pick(path)
            label.bind("<Button-3>", on_select)
        except Exception:
            ctk.CTkLabel(preview, text="Failed to load image").pack(expand=True)

        preview.bind("<Escape>", lambda e: preview.destroy())

    def _selection_pick(self, path: str):
        if self.selection_phase == "best":
            self.selected_best = path
            self.selection_phase = "worst"
            self._build_selection_ui()
        else:
            self._register_pair(self.selected_best, path)

    # ---- Common ----

    def _register_pair(self, chosen: str, rejected: str):
        self.results.setdefault(self.current_group_index, []).append({'chosen': chosen, 'rejected': rejected})
        self.current_remaining_images = [image for image in self.current_remaining_images if image not in {chosen, rejected}]
        self.pairs_created_in_group += 1

        if self.pairs_created_in_group < self.pairs_per_group and len(self.current_remaining_images) >= 2:
            if self.mode == "elo":
                self._start_elo_round()
            else:
                self._start_selection_round()
        else:
            self._advance_group()

    def _skip_group(self):
        self._advance_group()

    def _advance_group(self):
        self.current_group_index += 1
        if self.current_group_index >= len(self.groups):
            self._show_export()
        else:
            self._start_group_round()

    def _display_image(self, master, path: str, row: int, col: int):
        try:
            pil_img = Image.open(path)
            self.update_idletasks()
            win_w = self.winfo_width() or self.winfo_screenwidth()
            win_h = self.winfo_height() or self.winfo_screenheight()
            max_w = max(400, win_w // 2 - 40)
            max_h = max(400, win_h - 200)
            pil_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, size=pil_img.size)

            label = ctk.CTkLabel(master, text="", image=ctk_img)
            label.image = ctk_img  # prevent GC
            label.pil_image = pil_img  # prevent GC
            label.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        except Exception:
            ctk.CTkLabel(master, text=os.path.basename(path)).grid(row=row, column=col, padx=10, pady=10)

    def _show_export(self):
        for widget in self.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        total_pairs = sum(len(pairs) for pairs in self.results.values())
        groups_with_pairs = len(self.results)
        skipped = len(self.groups) - groups_with_pairs
        ctk.CTkLabel(frame, text="Scoring Complete!", font=("", 24, "bold")).pack(pady=(0, 20))
        summary = f"{total_pairs} chosen/rejected pairs from {groups_with_pairs} prompt groups ready to export."
        if skipped > 0:
            summary += f" ({skipped} groups skipped)"
        ctk.CTkLabel(frame, text=summary, font=("", 14)).pack(pady=(0, 20))

        val_frame = ctk.CTkFrame(frame, fg_color="transparent")
        val_frame.pack(pady=(0, 20))
        ctk.CTkLabel(val_frame, text="Validation %:").pack(side="left", padx=(0, 10))
        self.val_percentage_var = ctk.StringVar(value="10")
        ctk.CTkEntry(val_frame, textvariable=self.val_percentage_var, width=60).pack(side="left")
        ctk.CTkLabel(val_frame, text="(0 = no validation split)").pack(side="left", padx=(10, 0))

        ctk.CTkButton(frame, text="Export Chosen/Rejected Folders", width=300,
                      command=self._export).pack(pady=10)

    def _export(self):
        output_dir = filedialog.askdirectory(title="Select output directory for chosen/rejected folders")
        if not output_dir:
            return

        if has_existing_exports(output_dir):
            if not messagebox.askyesno("Overwrite?",
                                       "The selected directory already contains exported chosen/rejected pairs.\n"
                                       "Existing files with the same names will be overwritten.\n\n"
                                       "Continue?"):
                return

        try:
            val_pct = float(self.val_percentage_var.get())
        except (ValueError, AttributeError):
            val_pct = 0.0
        val_pct = max(0.0, min(100.0, val_pct))

        result = export_curated_pairs(self.groups, self.results, output_dir, val_percentage=val_pct)
        chosen_train, rejected_train, chosen_val, rejected_val, skipped, val_count, train_count = result
        total_pairs = sum(len(pairs) for pairs in self.results.values())
        msg = f"Exported {total_pairs} pairs ({train_count} train, {val_count} val)"
        if skipped > 0:
            msg += f", {skipped} groups skipped"
        msg += f"\n\nTrain:\n  {chosen_train}\n  {rejected_train}"
        if val_count > 0:
            msg += f"\n\nVal:\n  {chosen_val}\n  {rejected_val}"
        msg += f"\n\nConcept config saved to:\n  {os.path.join(output_dir, 'concepts.json')}"
        messagebox.showinfo("Export Complete", msg)
        self.destroy()
