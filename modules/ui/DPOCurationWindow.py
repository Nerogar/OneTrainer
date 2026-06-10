import os
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Empty, Full, Queue
from tkinter import filedialog, messagebox

from modules.util import path_util
from modules.util.dpo_curation_util import (
    export_single_pair,
    finalize_export,
    find_exported_file,
    find_orphaned_pairs,
    is_byte_identical_used,
    is_source_used,
    load_manifest,
    manifest_pair_counts,
    manifest_used_fingerprints,
    manifest_used_sources,
    prune_orphaned_pairs,
    remove_pair,
    resolve_aspect_ratio,
    walk_skipping_dotted,
)
from modules.util.dpo_swiss_service import SwissTournament, outermost_pairs
from modules.util.image_metadata_util import extract_metadata, strip_angle_bracket_segments
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk
from PIL import Image


class DPOCurationWindow(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("DPO Pair Tool")
        self.geometry("1400x900")
        self.resizable(True, True)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.current_remaining_images: list[str] = []
        self.pairs_per_group = 1
        self.pairs_created_in_group = 0

        self.source_folder: str | None = None
        self.output_dir: str | None = None
        self.manifest: dict = {"pairs": []}
        self.source_path_var = ctk.StringVar(value="(none selected)")
        self.output_path_var = ctk.StringVar(value="(none selected)")

        # Background worker state
        self._ready_queue: Queue = Queue(maxsize=10)
        self._worker_thread: threading.Thread | None = None
        self._worker_stop = threading.Event()
        self._scan_count = 0
        self._groups_queued = 0
        self._groups_shown = 0
        self._worker_finished = False
        self._current_group: dict | None = None

        # Tournament state
        self.swiss: SwissTournament | None = None
        self.swiss_match: tuple[str, str] | None = None
        self.ranked_order: list[str] = []
        self.rank_selected_idx: int | None = None
        self.swiss_export_pairs_var = ctk.StringVar(value="1")
        self._rank_badges: list = []
        self._thumb_cache: dict[str, tuple] = {}

        # Selection state
        self.selection_phase = "best"  # "best" or "worst"
        self.selected_best: str | None = None

        self.mode = "selection"  # "swiss" or "selection"
        self.pairs_per_group_var = ctk.StringVar(value="1")

        # Review mode state
        self._review_pairs: list[dict] = []
        self._review_index = 0
        self._review_removed = 0

        self._build_start_ui()

        self.after(200, lambda: set_window_icon(self))
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

    def _on_close(self):
        self._worker_stop.set()
        self.grab_release()
        self.destroy()

    def _safe_grab(self):
        try:
            self.grab_set()
            self.focus_set()
        except Exception:
            pass

    def _build_start_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(expand=True, fill="both", padx=40, pady=30)

        ctk.CTkLabel(frame, text="DPO Pair Tool", font=("", 28, "bold")).pack(pady=(0, 10))
        ctk.CTkLabel(
            frame, text="Curate chosen/rejected pairs from generated images.", font=("", 13), text_color="gray"
        ).pack(pady=(0, 25))

        # Folders section
        folder_card = ctk.CTkFrame(frame, border_width=1, border_color="gray30", corner_radius=8)
        folder_card.pack(fill="x", pady=(0, 15), padx=20)

        ctk.CTkLabel(folder_card, text="Folders", font=("", 13, "bold")).pack(anchor="w", padx=15, pady=(12, 8))

        src_frame = ctk.CTkFrame(folder_card, fg_color="transparent")
        src_frame.pack(fill="x", padx=15, pady=(0, 8))
        ctk.CTkButton(src_frame, text="Source Folder", width=150, command=self._select_source).pack(
            side="left", padx=(0, 10)
        )
        ctk.CTkLabel(src_frame, textvariable=self.source_path_var, anchor="w").pack(side="left", fill="x", expand=True)

        out_frame = ctk.CTkFrame(folder_card, fg_color="transparent")
        out_frame.pack(fill="x", padx=15, pady=(0, 12))
        ctk.CTkButton(out_frame, text="Output Folder", width=150, command=self._select_output).pack(
            side="left", padx=(0, 10)
        )
        ctk.CTkLabel(out_frame, textvariable=self.output_path_var, anchor="w").pack(side="left", fill="x", expand=True)

        self.resume_label = ctk.CTkLabel(frame, text="", font=("", 12), text_color="green")
        self.resume_label.pack(pady=(0, 10))

        # Settings section
        settings_card = ctk.CTkFrame(frame, border_width=1, border_color="gray30", corner_radius=8)
        settings_card.pack(fill="x", pady=(0, 20), padx=20)

        pair_count_frame = ctk.CTkFrame(settings_card, fg_color="transparent")
        pair_count_frame.pack(padx=15, pady=12)
        ctk.CTkLabel(pair_count_frame, text="Pairs per group:").pack(side="left", padx=(0, 10))
        ctk.CTkEntry(pair_count_frame, textvariable=self.pairs_per_group_var, width=60).pack(side="left")
        ctk.CTkLabel(
            pair_count_frame, text="How many pairs to collect before moving on.", anchor="w", text_color="gray"
        ).pack(side="left", padx=(10, 0))

        # Action buttons
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(pady=(0, 10))
        ctk.CTkButton(btn_frame, text="Start (Tournament)", width=250, command=lambda: self._start("swiss")).pack(
            side="left", padx=10
        )
        ctk.CTkButton(btn_frame, text="Start (Selection)", width=250, command=lambda: self._start("selection")).pack(
            side="left", padx=10
        )

        ctk.CTkButton(frame, text="Review Pairs", width=250, fg_color="gray40", command=self._start_review).pack(
            pady=(5, 0)
        )

    def _select_source(self):
        self.grab_release()
        folder = filedialog.askdirectory(title="Select folder of generated images")
        try:
            self.grab_set()
        except Exception:
            self.after(100, self._safe_grab)
        self.focus_set()
        if folder:
            self.source_folder = folder
            self.source_path_var.set(folder)

    def _select_output(self):
        self.grab_release()
        folder = filedialog.askdirectory(title="Select output directory for exports")
        try:
            self.grab_set()
        except Exception:
            self.after(100, self._safe_grab)
        self.focus_set()
        if folder:
            self.output_dir = folder
            self.output_path_var.set(folder)
            self.manifest = load_manifest(folder)
            pruned = prune_orphaned_pairs(folder, self.manifest)
            existing = len(self.manifest.get("pairs", []))
            if existing:
                msg = f"Found {existing} existing pairs \u2014 completed groups will be skipped."
                if pruned:
                    msg += f"\n(Removed {pruned} orphaned entries with missing files.)"
                self.resume_label.configure(text=msg)
            else:
                if pruned:
                    self.resume_label.configure(text=f"Removed {pruned} orphaned entries with missing files.")
                else:
                    self.resume_label.configure(text="")

    def _start(self, mode: str):
        if not self.source_folder:
            messagebox.showwarning("No Source", "Select a source folder first.")
            return
        if not self.output_dir:
            messagebox.showwarning("No Output", "Select an output folder first.")
            return

        self.mode = mode
        self.pairs_per_group = self._parse_pairs_per_group()
        self._groups_shown = 0
        self._groups_queued = 0
        self._scan_count = 0
        self._worker_finished = False
        self._ready_queue = Queue(maxsize=10)
        self._worker_stop = threading.Event()

        self._worker_thread = threading.Thread(
            target=self._background_scan_and_dedup,
            args=(self.source_folder,),
            daemon=True,
        )
        self._worker_thread.start()
        self._show_scanning_ui()

    @staticmethod
    def _dhash(path: str, hash_size: int = 8) -> int:
        """Compute a difference hash for duplicate detection."""
        with Image.open(path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
        bits = 0
        for row in range(hash_size):
            for col in range(hash_size):
                idx = row * (hash_size + 1) + col
                if pixels[idx] < pixels[idx + 1]:
                    bits |= 1 << (row * hash_size + col)
        return bits

    @staticmethod
    def _dedup_by_dhash(images: list[str]) -> list[str]:
        """Remove visually identical images from a list, keeping the latest mtime of each."""
        seen: dict[int, int] = {}
        unique: list[str] = []
        for path in images:
            try:
                h = DPOCurationWindow._dhash(path)
            except Exception:
                unique.append(path)
                continue
            if h not in seen:
                seen[h] = len(unique)
                unique.append(path)
            else:
                idx = seen[h]
                try:
                    if os.path.getmtime(path) > os.path.getmtime(unique[idx]):
                        unique[idx] = path
                except OSError:
                    continue
        return unique

    @staticmethod
    def _extract_group_key(path: str) -> tuple[str, str]:
        """Run metadata extraction for one file and reduce it to the (prompt, aspectratio)
        bucket key. The aspect component is the trainer bucket label (e.g. "7:4"), derived
        from actual pixel dimensions — so 1344x768 and 1680x960 (both ~16:9 presets) group
        together, exactly as AspectBucketing crops them to the same bucket at train time.
        Safe to call from a worker thread."""
        meta = extract_metadata(path)
        prompt = meta.get("prompt", "").strip()
        ar = resolve_aspect_ratio(meta.get("aspectratio", ""), path)
        if prompt:
            prompt = strip_angle_bracket_segments(prompt)
        if not prompt:
            prompt = "UNCONDITIONAL"
        return prompt, ar

    @staticmethod
    def _scan_worker_count() -> int:
        # Metadata extraction is I/O-bound (file open + small read + parse), so oversubscribing
        # the CPU pays off — but cap it so we don't thrash on HDDs or hit Windows' per-process
        # thread limits.
        return min(16, (os.cpu_count() or 4) * 2)

    def _background_scan_and_dedup(self, folder: str):
        supported = path_util.supported_image_extensions()

        # Pass 1: enumerate candidate paths up front. os.walk uses scandir under the hood
        # so this is cheap even on large trees, and collecting paths first lets us fan
        # the expensive per-file metadata reads out across a thread pool. Dot-prefixed
        # subdirectories (.thumbnails, .cache, ...) are pruned from the walk entirely.
        candidate_paths: list[str] = []
        for root, files in walk_skipping_dotted(folder):
            if self._worker_stop.is_set():
                return
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported:
                    candidate_paths.append(os.path.join(root, filename))

        # Pass 2: parallel metadata extraction. The grouping dict is mutated only on the
        # consumer side of `as_completed`, so no lock is needed; `_scan_count` is a plain
        # int updated here too (single writer).
        groups_dict: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
        if candidate_paths:
            with ThreadPoolExecutor(max_workers=self._scan_worker_count()) as pool:
                future_to_path = {pool.submit(self._extract_group_key, p): p for p in candidate_paths}
                for future in as_completed(future_to_path):
                    if self._worker_stop.is_set():
                        # Don't wait for in-flight futures on cancel — each is a single
                        # file read and will finish on its own. cancel_futures handles
                        # the rest of the queue.
                        pool.shutdown(wait=False, cancel_futures=True)
                        return
                    try:
                        prompt, ar = future.result()
                    except Exception:
                        self._scan_count += 1
                        continue
                    groups_dict[(prompt, ar)].append(future_to_path[future])
                    self._scan_count += 1

        raw_groups = [
            {"prompt": prompt, "aspectratio": ar, "images": images}
            for (prompt, ar), images in groups_dict.items()
            if len(images) >= 2
        ]
        random.shuffle(raw_groups)

        existing_counts = manifest_pair_counts(self.manifest)
        # Path-based filter is cheap; fingerprint filter backfills SHA-256 for legacy
        # entries (one-shot disk read per old pair) then uses size as a pre-filter so
        # candidates only get hashed when their byte-length matches an existing pair.
        used_sources = manifest_used_sources(self.manifest)
        used_fingerprints = manifest_used_fingerprints(self.manifest, self.output_dir)

        for group in raw_groups:
            if self._worker_stop.is_set():
                return

            group_key = (group["prompt"], group["aspectratio"])
            is_unconditional = group["prompt"] == "UNCONDITIONAL"
            if not is_unconditional and existing_counts.get(group_key, 0) >= self.pairs_per_group:
                continue

            filtered: list[str] = []
            for path in group["images"]:
                if is_source_used(used_sources, path):
                    continue
                if is_byte_identical_used(used_fingerprints, path):
                    continue
                filtered.append(path)
            if len(filtered) < 2:
                continue
            group["images"] = filtered

            deduped = self._dedup_by_dhash(group["images"])
            if len(deduped) >= 2:
                group["images"] = deduped
                self._groups_queued += 1
                while not self._worker_stop.is_set():
                    try:
                        self._ready_queue.put(group, timeout=0.5)
                        break
                    except Full:
                        continue

        self._worker_finished = True

    def _show_scanning_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(expand=True)

        ctk.CTkLabel(frame, text="Scanning...", font=("", 20, "bold")).pack(pady=(0, 15))
        self._scan_progress = ctk.CTkProgressBar(frame, width=300, mode="indeterminate")
        self._scan_progress.pack(pady=(0, 15))
        self._scan_progress.start()
        self._scan_label = ctk.CTkLabel(frame, text="0 files scanned", font=("", 14))
        self._scan_label.pack()

        self._poll_scan()

    def _poll_scan(self):
        if self._worker_stop.is_set():
            return

        self._scan_label.configure(text=f"{self._scan_count} files scanned")

        if not self._ready_queue.empty():
            self._next_group()
        elif self._worker_finished:
            if self._groups_queued == 0:
                messagebox.showwarning("No Groups", "No images with extractable prompt metadata found.")
                self._build_start_ui()
            else:
                self._show_export()
        else:
            self.after(100, self._poll_scan)

    def _parse_pairs_per_group(self) -> int:
        try:
            value = int(str(self.pairs_per_group_var.get()).strip())
        except (TypeError, ValueError):
            return 1
        return max(1, value)

    def _next_group(self):
        while True:
            try:
                group = self._ready_queue.get_nowait()
            except Empty:
                if self._worker_finished:
                    self._show_export()
                    return
                self._show_waiting_ui()
                return

            existing_counts = manifest_pair_counts(self.manifest)
            group_key = (group["prompt"], group["aspectratio"])
            pairs_done = existing_counts.get(group_key, 0)
            is_unconditional = group["prompt"] == "UNCONDITIONAL"
            if not is_unconditional and pairs_done >= self.pairs_per_group:
                continue

            self._current_group = group
            self._groups_shown += 1
            self.pairs_created_in_group = pairs_done
            self.current_remaining_images = list(group["images"])

            if self.mode == "swiss":
                self._start_swiss_round()
            else:
                self._start_selection_round()
            return

    def _show_waiting_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(expand=True)

        ctk.CTkLabel(frame, text="Preparing next group...", font=("", 18)).pack()
        self._poll_waiting()

    def _poll_waiting(self):
        if self._worker_stop.is_set():
            return
        if not self._ready_queue.empty() or self._worker_finished:
            self._next_group()
        else:
            self.after(100, self._poll_waiting)

    # ---- Tournament Mode ----

    def _start_swiss_round(self):
        if len(self.current_remaining_images) < 2:
            self._advance_group()
            return
        self.swiss = SwissTournament(self.current_remaining_images)
        self._thumb_cache = {}
        self._swiss_next_match()

    def _swiss_next_match(self):
        match = self.swiss.next_match()
        if match is None:
            self._start_ranked_review()
            return
        self.swiss_match = match
        self._build_swiss_ui()

    def _build_swiss_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        group = self._current_group

        # Header
        self._build_prompt_expander(self, group["prompt"])

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(header, text=f"AR: {group['aspectratio']}", font=("", 12)).pack(side="left", padx=15)
        ctk.CTkLabel(header, text=f"Group {self._groups_shown} / {self._groups_queued}", font=("", 12)).pack(
            side="left", padx=15
        )
        ctk.CTkLabel(
            header,
            text=f"Round {self.swiss.current_round}/{self.swiss.total_rounds}"
            f" — Match {self.swiss.matches_played() + 1}/{self.swiss.matches_total()}",
            font=("", 12, "bold"),
        ).pack(side="left", padx=15)

        # Images
        img_frame = ctk.CTkFrame(self, fg_color="transparent")
        img_frame.pack(expand=True, fill="both", padx=10, pady=5)
        img_frame.grid_columnconfigure(0, weight=1)
        img_frame.grid_columnconfigure(1, weight=1)
        img_frame.grid_rowconfigure(0, weight=1)

        for col, path in enumerate(self.swiss_match):
            self._display_image(img_frame, path, row=0, col=col)
            player = self.swiss.player(path)
            ctk.CTkLabel(img_frame, text=f"Score {player.score:g} · Elo {player.elo:.0f}", font=("", 12)).grid(
                row=1, column=col, pady=(0, 5)
            )

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(btn_frame, text="A is Better (←)", width=200, command=lambda: self._swiss_vote("a")).pack(
            side="left", padx=20, expand=True
        )
        ctk.CTkButton(btn_frame, text="Tie (↓)", width=200, command=lambda: self._swiss_vote("tie")).pack(
            side="left", padx=20, expand=True
        )
        ctk.CTkButton(btn_frame, text="B is Better (→)", width=200, command=lambda: self._swiss_vote("b")).pack(
            side="left", padx=20, expand=True
        )
        ctk.CTkButton(btn_frame, text="Skip Group", width=150, fg_color="#8B4513", command=self._skip_group).pack(
            side="right", padx=10
        )
        ctk.CTkButton(
            btn_frame, text="Finish Early", width=150, fg_color="gray", command=self._start_ranked_review
        ).pack(side="right", padx=10)

        # Keyboard bindings
        self.bind("<Left>", lambda e: self._swiss_vote("a"))
        self.bind("<Right>", lambda e: self._swiss_vote("b"))
        self.bind("<Down>", lambda e: self._swiss_vote("tie"))

    def _swiss_vote(self, winner: str):
        a, b = self.swiss_match
        self.swiss.report(a, b, winner)
        self._swiss_next_match()

    # ---- Ranked Review (after the tournament rounds) ----

    def _start_ranked_review(self):
        self.ranked_order = self.swiss.standings()
        self.rank_selected_idx = None
        max_pairs = len(self.ranked_order) // 2
        default_pairs = min(max(1, self.pairs_per_group - self.pairs_created_in_group), max_pairs)
        self.swiss_export_pairs_var.set(str(default_pairs))
        self._build_ranked_review_ui()

    def _parse_export_pairs(self) -> int:
        try:
            value = int(str(self.swiss_export_pairs_var.get()).strip())
        except (TypeError, ValueError):
            value = 1
        return max(0, min(value, len(self.ranked_order) // 2))

    def _build_ranked_review_ui(self):
        for widget in self.winfo_children():
            widget.destroy()
        for key in ("<Left>", "<Right>", "<Down>"):
            self.unbind(key)

        group = self._current_group

        # Header
        self._build_prompt_expander(self, group["prompt"])

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(header, text=f"AR: {group['aspectratio']}", font=("", 12)).pack(side="left", padx=15)
        ctk.CTkLabel(header, text=f"Group {self._groups_shown} / {self._groups_queued}", font=("", 12)).pack(
            side="left", padx=15
        )
        ctk.CTkLabel(
            header,
            text="Ranked best → worst. Left-click two images to swap them; right-click to preview.",
            font=("", 13, "bold"),
        ).pack(side="right", padx=15)

        # Ranked thumbnail grid
        grid_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        grid_frame.pack(expand=True, fill="both", padx=10, pady=5)

        cols = max(1, min(5, int(self.winfo_width() / 300) or 4))
        self._rank_badges = []
        for index, path in enumerate(self.ranked_order):
            row, col = divmod(index, cols)
            grid_frame.grid_columnconfigure(col, weight=1)
            selected = index == self.rank_selected_idx
            tile = ctk.CTkFrame(
                grid_frame,
                fg_color="transparent",
                border_width=3 if selected else 0,
                border_color="#1F6AA5",
            )
            tile.grid(row=row, column=col, padx=5, pady=5)

            player = self.swiss.player(path)
            badge = ctk.CTkLabel(tile, text=f"#{index + 1} · {player.score:g} pts", font=("", 12, "bold"))
            badge.pack(pady=(4, 0))
            self._rank_badges.append(badge)

            thumb = self._ranked_thumbnail(tile, path)
            thumb.pack(padx=4, pady=4)
            for widget in (badge, thumb):
                widget.bind("<Button-1>", lambda e, i=index: self._rank_click(i))
                widget.bind("<Button-3>", lambda e, p=path: self._selection_preview(p, pick=False))

        self._update_rank_badges()

        # Footer
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(footer, text="Pairs to export:").pack(side="left", padx=(15, 10))
        pairs_entry = ctk.CTkEntry(footer, textvariable=self.swiss_export_pairs_var, width=60)
        pairs_entry.pack(side="left")
        pairs_entry.bind("<KeyRelease>", lambda e: self._update_rank_badges())
        ctk.CTkLabel(
            footer, text="Pair 1 = green #1 vs red last, pair 2 = next pair inward, ...", text_color="gray"
        ).pack(side="left", padx=(10, 0))
        ctk.CTkButton(footer, text="Skip Group", width=150, fg_color="#8B4513", command=self._skip_group).pack(
            side="right", padx=10
        )
        ctk.CTkButton(footer, text="Export Pairs", width=200, command=self._export_ranked_pairs).pack(
            side="right", padx=10
        )

    def _ranked_thumbnail(self, master, path: str):
        thumb_size = 220
        cached = self._thumb_cache.get(path)
        if cached is None:
            try:
                with Image.open(path) as _raw:
                    pil_img = self._fit_image(_raw, thumb_size, thumb_size).copy()
                cached = (ctk.CTkImage(light_image=pil_img, size=pil_img.size), pil_img)
            except Exception:
                cached = (None, None)
            self._thumb_cache[path] = cached
        ctk_img, pil_img = cached
        if ctk_img is None:
            return ctk.CTkLabel(master, text=os.path.basename(path))
        label = ctk.CTkLabel(master, text="", image=ctk_img)
        label.image = ctk_img  # prevent GC
        label.pil_image = pil_img  # prevent GC - CTkImage needs the PIL data alive for DPI re-rendering
        return label

    def _update_rank_badges(self):
        pair_count = self._parse_export_pairs()
        total = len(self._rank_badges)
        for index, badge in enumerate(self._rank_badges):
            if index < pair_count:
                badge.configure(text_color="#2FA572")  # exported as chosen
            elif index >= total - pair_count:
                badge.configure(text_color="#C84B4B")  # exported as rejected
            else:
                badge.configure(text_color=("gray10", "gray90"))

    def _rank_click(self, index: int):
        if self.rank_selected_idx is None:
            self.rank_selected_idx = index
        elif self.rank_selected_idx == index:
            self.rank_selected_idx = None
        else:
            order = self.ranked_order
            order[self.rank_selected_idx], order[index] = order[index], order[self.rank_selected_idx]
            self.rank_selected_idx = None
        self._build_ranked_review_ui()

    def _export_ranked_pairs(self):
        pairs = outermost_pairs(self.ranked_order, self._parse_export_pairs())
        if not pairs:
            self._advance_group()
            return

        group = self._current_group
        exported: set[str] = set()
        for chosen, rejected in pairs:
            export_single_pair(
                self.output_dir,
                self.manifest,
                chosen,
                rejected,
                group["prompt"],
                group["aspectratio"],
            )
            exported |= {chosen, rejected}

        self.current_remaining_images = [
            image for image in self.current_remaining_images if image not in exported
        ]
        self.pairs_created_in_group += len(pairs)
        self._advance_group()

    # ---- Selection Mode ----

    def _start_selection_round(self):
        self.selection_phase = "best"
        self.selected_best = None
        self._build_selection_ui()

    def _build_selection_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        group = self._current_group

        # Header
        self._build_prompt_expander(self, group["prompt"])

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(header, text=f"AR: {group['aspectratio']}", font=("", 12)).pack(side="left", padx=15)
        ctk.CTkLabel(header, text=f"Group {self._groups_shown} / {self._groups_queued}", font=("", 12)).pack(
            side="left", padx=15
        )

        ctk.CTkButton(header, text="Skip Group", width=120, fg_color="#8B4513", command=self._skip_group).pack(
            side="right", padx=10
        )

        if self.selection_phase == "best":
            phase_text = (
                f"Pair {self.pairs_created_in_group + 1}/{self.pairs_per_group}. "
                "Click to view, then right-click to select as BEST"
            )
        else:
            best_name = os.path.basename(self.selected_best) if self.selected_best else "?"
            phase_text = (
                f"Pair {self.pairs_created_in_group + 1}/{self.pairs_per_group}. "
                f"Best: {best_name}. Now right-click to select WORST"
            )
        ctk.CTkLabel(
            header,
            text=phase_text,
            font=("", 14, "bold"),
            text_color="green" if self.selection_phase == "best" else "red",
        ).pack(side="right", padx=15)

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
            with Image.open(path) as _raw:
                pil_img = self._fit_image(_raw, thumb_size, thumb_size).copy()
            ctk_img = ctk.CTkImage(light_image=pil_img, size=pil_img.size)

            label = ctk.CTkLabel(master, text="", image=ctk_img)
            label.image = ctk_img  # prevent GC
            label.pil_image = pil_img  # prevent GC - CTkImage needs the PIL data alive for DPI re-rendering
            label.grid(row=row, column=col, padx=5, pady=5)
            label.bind("<Button-1>", lambda e, p=path: self._selection_preview(p))
            label.bind("<Button-3>", lambda e, p=path: self._selection_pick(p))
        except Exception:
            ctk.CTkLabel(master, text=os.path.basename(path)).grid(row=row, column=col, padx=5, pady=5)

    def _selection_preview(self, path: str, pick: bool = True):
        preview = ctk.CTkToplevel(self)
        preview.title("Preview — Right-click to select" if pick else "Preview — Esc to close")
        preview.attributes("-fullscreen", True)
        preview.transient(self)
        preview.wait_visibility()
        preview.grab_set()
        preview.focus_set()

        try:
            sw, sh = preview.winfo_screenwidth(), preview.winfo_screenheight()
            with Image.open(path) as _raw:
                pil_img = self._fit_image(_raw, sw, sh - 50).copy()
            ctk_img = ctk.CTkImage(light_image=pil_img, size=pil_img.size)

            label = ctk.CTkLabel(preview, text="", image=ctk_img)
            label.image = ctk_img  # prevent GC
            label.pil_image = pil_img  # prevent GC
            label.pack(expand=True)

            def on_select(e):
                preview.destroy()
                self._selection_pick(path)

            if pick:
                label.bind("<Button-3>", on_select)
            else:
                label.bind("<Button-3>", lambda e: preview.destroy())
        except Exception:
            ctk.CTkLabel(preview, text="Failed to load image").pack(expand=True)

        preview.bind("<Escape>", lambda e: preview.destroy())

    def _selection_pick(self, path: str):
        if self.selection_phase == "best":
            self.selected_best = path
            self.selection_phase = "worst"
            self._build_selection_ui()
        else:
            remaining_after = [img for img in self.current_remaining_images if img not in {self.selected_best, path}]
            can_continue = len(remaining_after) >= 2

            if can_continue:
                result = messagebox.askyesnocancel(
                    "Pair Selected",
                    f"Best: {os.path.basename(self.selected_best)}\n"
                    f"Worst: {os.path.basename(path)}\n\n"
                    f"Yes = Accept & keep scoring this prompt\n"
                    f"No = Accept & move to next group\n"
                    f"Cancel = Don't accept",
                )
                if result is None:
                    return
                self._register_pair(self.selected_best, path, continue_scoring=result)
            else:
                self._register_pair(self.selected_best, path)

    # ---- Common ----

    def _register_pair(self, chosen: str, rejected: str, continue_scoring: bool = False):
        group = self._current_group
        export_single_pair(
            self.output_dir,
            self.manifest,
            chosen,
            rejected,
            group["prompt"],
            group["aspectratio"],
        )
        self.current_remaining_images = [
            image for image in self.current_remaining_images if image not in {chosen, rejected}
        ]
        self.pairs_created_in_group += 1

        is_unconditional = group["prompt"] == "UNCONDITIONAL"
        keep_going = continue_scoring or is_unconditional or self.pairs_created_in_group < self.pairs_per_group
        if keep_going and len(self.current_remaining_images) >= 2:
            if self.mode == "swiss":
                self._start_swiss_round()
            else:
                self._start_selection_round()
        else:
            self._advance_group()

    def _skip_group(self):
        self._advance_group()

    def _advance_group(self):
        self._next_group()

    def _fit_image(self, pil_img: Image.Image, max_w: int, max_h: int) -> Image.Image:
        scale = min(max_w / pil_img.width, max_h / pil_img.height)
        new_w = max(1, int(pil_img.width * scale))
        new_h = max(1, int(pil_img.height * scale))
        return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _build_prompt_expander(self, parent, prompt: str):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=(5, 0))

        if prompt == "UNCONDITIONAL":
            ctk.CTkLabel(
                frame,
                text="UNCONDITIONAL",
                font=("", 14, "bold"),
                text_color="#FFD700",
                fg_color="#3A3000",
                corner_radius=4,
            ).pack(side="left", padx=(0, 8), ipadx=8, ipady=2)
            return frame

        truncated = prompt[:100] + ("..." if len(prompt) > 100 else "")
        expanded = ctk.BooleanVar(value=False)

        toggle_btn = ctk.CTkButton(
            frame, text="Prompt [+]", width=90, height=24, font=("", 11), fg_color="gray30", command=lambda: _toggle()
        )
        toggle_btn.pack(side="left", padx=(0, 8))

        text_label = ctk.CTkLabel(frame, text=truncated, font=("", 12), anchor="w", wraplength=0)
        text_label.pack(side="left", fill="x", expand=True)

        def _toggle():
            if expanded.get():
                expanded.set(False)
                toggle_btn.configure(text="Prompt [+]")
                text_label.configure(text=truncated, wraplength=0)
            else:
                expanded.set(True)
                toggle_btn.configure(text="Prompt [-]")
                text_label.configure(text=prompt, wraplength=max(200, frame.winfo_width() - 110))

        return frame

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
            label.image = ctk_img  # prevent GC
            label.pil_image = pil_img  # prevent GC
            label.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        except Exception:
            ctk.CTkLabel(master, text=os.path.basename(path)).grid(row=row, column=col, padx=10, pady=10)

    def _show_export(self):
        for widget in self.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(expand=True, fill="both", padx=40, pady=30)

        total_pairs = len(self.manifest.get("pairs", []))
        unique_groups = len({(e["prompt"], e.get("aspectratio", "")) for e in self.manifest.get("pairs", [])})
        skipped = max(0, self._groups_queued - unique_groups)

        ctk.CTkLabel(frame, text="Scoring Complete", font=("", 28, "bold")).pack(pady=(0, 20))

        # Summary card
        summary_card = ctk.CTkFrame(frame, border_width=1, border_color="gray30", corner_radius=8)
        summary_card.pack(fill="x", pady=(0, 20), padx=20)

        summary = f"{total_pairs} chosen/rejected pairs from {unique_groups} prompt groups exported."
        if skipped > 0:
            summary += f"  ({skipped} groups skipped)"
        ctk.CTkLabel(summary_card, text=summary, font=("", 14)).pack(padx=15, pady=(12, 5))
        ctk.CTkLabel(summary_card, text=self.output_dir, font=("", 11), text_color="gray").pack(padx=15, pady=(0, 12))

        val_frame = ctk.CTkFrame(frame, fg_color="transparent")
        val_frame.pack(pady=(0, 20))
        ctk.CTkLabel(val_frame, text="Validation %:").pack(side="left", padx=(0, 10))
        self.val_percentage_var = ctk.StringVar(value="10")
        ctk.CTkEntry(val_frame, textvariable=self.val_percentage_var, width=60).pack(side="left")
        ctk.CTkLabel(val_frame, text="(0 = no validation split)", text_color="gray").pack(side="left", padx=(10, 0))

        ctk.CTkButton(frame, text="Finalize (Train/Val Split + Concepts)", width=350, command=self._finalize).pack(
            pady=10
        )
        ctk.CTkButton(frame, text="Close (Pairs Already Saved)", width=350, fg_color="gray", command=self.destroy).pack(
            pady=10
        )

    def _finalize(self):
        try:
            val_pct = float(self.val_percentage_var.get())
        except (ValueError, AttributeError):
            val_pct = 0.0
        val_pct = max(0.0, min(100.0, val_pct))

        train_count, val_count = finalize_export(self.output_dir, self.manifest, val_percentage=val_pct)
        total_pairs = len(self.manifest.get("pairs", []))
        msg = f"Finalized {total_pairs} pairs ({train_count} train, {val_count} val)"
        msg += f"\n\nConcept config saved to:\n  {os.path.join(self.output_dir, 'concepts.json')}"
        messagebox.showinfo("Finalize Complete", msg)
        self.destroy()

    # ---- Review Pairs Mode ----

    def _start_review(self):
        if not self.output_dir:
            messagebox.showwarning("No Output", "Select an output folder first.")
            return

        self.manifest = load_manifest(self.output_dir)
        pairs = self.manifest.get("pairs", [])
        if not pairs:
            messagebox.showinfo("No Pairs", "No pairs found in the manifest.")
            return

        orphans = find_orphaned_pairs(self.output_dir, self.manifest)
        if orphans:
            result = messagebox.askyesno(
                "Orphaned Pairs Found",
                f"Found {len(orphans)} pair(s) with missing files.\n\nRemove them from the manifest?",
            )
            if result:
                for entry in orphans:
                    remove_pair(self.output_dir, self.manifest, entry)
                pairs = self.manifest.get("pairs", [])
                if not pairs:
                    messagebox.showinfo("No Pairs", "All pairs were orphaned and removed.")
                    return

        self._review_pairs = list(pairs)
        self._review_index = 0
        self._review_removed = 0
        self._build_review_ui()

    def _build_review_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        if self._review_index >= len(self._review_pairs):
            self._show_review_summary()
            return

        entry = self._review_pairs[self._review_index]
        total = len(self._review_pairs)

        # Header
        self._build_prompt_expander(self, entry.get("prompt", ""))

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(header, text=f"Pair {self._review_index + 1} / {total}", font=("", 14, "bold")).pack(
            side="left", padx=15
        )
        ctk.CTkLabel(header, text=f"AR: {entry.get('aspectratio', '')}", font=("", 12)).pack(side="left", padx=15)
        ctk.CTkLabel(header, text=f"Removed: {self._review_removed}", font=("", 12), text_color="red").pack(
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

        chosen_dir = os.path.join(self.output_dir, "chosen")
        rejected_dir = os.path.join(self.output_dir, "rejected")
        chosen_path = find_exported_file(chosen_dir, entry["chosen_file"])
        rejected_path = find_exported_file(rejected_dir, entry["rejected_file"])

        if chosen_path:
            self._display_image(img_frame, chosen_path, row=1, col=0)
        else:
            ctk.CTkLabel(img_frame, text="(missing)", font=("", 14), text_color="gray").grid(row=1, column=0)
        if rejected_path:
            self._display_image(img_frame, rejected_path, row=1, col=1)
        else:
            ctk.CTkLabel(img_frame, text="(missing)", font=("", 14), text_color="gray").grid(row=1, column=1)

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(
            btn_frame,
            text="← Back",
            width=150,
            command=lambda: self._review_advance(-1),
            state="normal" if self._review_index > 0 else "disabled",
        ).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Remove", width=150, fg_color="#B22222", command=self._review_remove).pack(
            side="left", padx=10, expand=True
        )
        ctk.CTkButton(btn_frame, text="Keep →", width=150, command=lambda: self._review_advance(1)).pack(
            side="right", padx=10
        )

        # Keyboard bindings
        self.bind("<Left>", lambda e: self._review_advance(-1) if self._review_index > 0 else None)
        self.bind("<Right>", lambda e: self._review_advance(1))
        self.bind("<Delete>", lambda e: self._review_remove())

    def _review_advance(self, delta: int):
        self._review_index += delta
        if self._review_index < 0:
            self._review_index = 0
        self._build_review_ui()

    def _review_remove(self):
        entry = self._review_pairs[self._review_index]
        remove_pair(self.output_dir, self.manifest, entry)
        self._review_pairs.pop(self._review_index)
        self._review_removed += 1
        if self._review_index >= len(self._review_pairs):
            self._review_index = max(0, len(self._review_pairs) - 1)
        self._build_review_ui()

    def _show_review_summary(self):
        for widget in self.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(frame, text="Review Complete", font=("", 24, "bold")).pack(pady=(0, 20))

        remaining = len(self.manifest.get("pairs", []))
        ctk.CTkLabel(frame, text=f"Kept: {remaining} pairs", font=("", 14)).pack(pady=5)
        ctk.CTkLabel(frame, text=f"Removed: {self._review_removed} pairs", font=("", 14), text_color="red").pack(pady=5)

        ctk.CTkButton(frame, text="Back to Start", width=250, command=self._build_start_ui).pack(pady=20)
        ctk.CTkButton(frame, text="Close", width=250, fg_color="gray", command=self.destroy).pack(pady=5)
