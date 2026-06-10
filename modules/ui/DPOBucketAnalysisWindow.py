import threading
from tkinter import filedialog, messagebox

from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_bucket_analysis_util import (
    analyze_concept,
    parse_target_resolutions,
    quantization_for_model,
)
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk


class DPOBucketAnalysisWindow(ctk.CTkToplevel):
    def __init__(self, parent, train_config: TrainConfig, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("DPO Bucket / Batch-Size Analyzer")
        self.geometry("1100x780")
        self.resizable(True, True)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.train_config = train_config

        self.concept_path: str | None = None
        self.concept_path_var = ctk.StringVar(value="(none selected)")
        self.batch_size_var = ctk.StringVar(value=str(max(1, int(getattr(train_config, "batch_size", 1) or 1))))

        resolution_str = getattr(train_config, "resolution", "") or ""
        parsed = parse_target_resolutions(resolution_str)
        if not parsed:
            parsed = [512]
        target_options = sorted({*parsed, 512, 768, 1024})
        default_target = str(min(parsed))
        self.target_options = [str(v) for v in target_options]
        self.target_var = ctk.StringVar(value=default_target)

        model_default_q = quantization_for_model(getattr(train_config, "model_type", ""))
        self.quant_options = ["8", "16", "32", "64", "128"]
        if str(model_default_q) not in self.quant_options:
            self.quant_options.append(str(model_default_q))
            self.quant_options = sorted(self.quant_options, key=int)
        self.quant_var = ctk.StringVar(value=str(model_default_q))

        self.status_var = ctk.StringVar(value="")
        self._analysis_thread: threading.Thread | None = None
        self._result: dict | None = None

        self._build_ui()

        self.after(200, lambda: set_window_icon(self))
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

    def _on_close(self):
        self.grab_release()
        self.destroy()

    def _build_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        root = ctk.CTkFrame(self, fg_color="transparent")
        root.pack(expand=True, fill="both", padx=20, pady=15)

        ctk.CTkLabel(root, text="DPO Bucket / Batch-Size Analyzer", font=("", 22, "bold")).pack(pady=(0, 6))
        ctk.CTkLabel(
            root,
            text="Select a chosen-side concept folder. Shows how many pairs each aspect "
            "bucket holds and what to add or remove for clean batches.",
            font=("", 12),
            text_color="gray",
            wraplength=1000,
            justify="left",
        ).pack(pady=(0, 15))

        folder_card = ctk.CTkFrame(root, border_width=1, border_color="gray30", corner_radius=8)
        folder_card.pack(fill="x", pady=(0, 10), padx=10)
        ctk.CTkLabel(folder_card, text="Chosen-side concept folder", font=("", 13, "bold")).pack(
            anchor="w", padx=15, pady=(12, 4)
        )
        folder_row = ctk.CTkFrame(folder_card, fg_color="transparent")
        folder_row.pack(fill="x", padx=15, pady=(0, 12))
        ctk.CTkButton(folder_row, text="Select Folder", width=150, command=self._select_folder).pack(
            side="left", padx=(0, 10)
        )
        ctk.CTkLabel(folder_row, textvariable=self.concept_path_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        settings_card = ctk.CTkFrame(root, border_width=1, border_color="gray30", corner_radius=8)
        settings_card.pack(fill="x", pady=(0, 10), padx=10)

        settings_row = ctk.CTkFrame(settings_card, fg_color="transparent")
        settings_row.pack(fill="x", padx=15, pady=12)

        ctk.CTkLabel(settings_row, text="Batch size:").pack(side="left", padx=(0, 6))
        ctk.CTkEntry(settings_row, textvariable=self.batch_size_var, width=70).pack(side="left", padx=(0, 20))

        ctk.CTkLabel(settings_row, text="Target resolution:").pack(side="left", padx=(0, 6))
        ctk.CTkOptionMenu(settings_row, variable=self.target_var, values=self.target_options, width=90).pack(
            side="left", padx=(0, 20)
        )

        ctk.CTkLabel(settings_row, text="Quantization:").pack(side="left", padx=(0, 6))
        ctk.CTkOptionMenu(settings_row, variable=self.quant_var, values=self.quant_options, width=80).pack(
            side="left", padx=(0, 20)
        )

        ctk.CTkButton(settings_row, text="Analyze", width=140, command=self._run_analysis).pack(side="right")

        self.results_frame = ctk.CTkScrollableFrame(root, fg_color="transparent")
        self.results_frame.pack(expand=True, fill="both", padx=10, pady=(0, 10))

        self._build_empty_results()

        ctk.CTkLabel(root, textvariable=self.status_var, font=("", 12), text_color="gray", anchor="w").pack(
            fill="x", padx=10, pady=(0, 4)
        )

    def _build_empty_results(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        ctk.CTkLabel(
            self.results_frame, text="Pick a folder and click Analyze to see the bucket breakdown.", text_color="gray"
        ).pack(pady=40)

    def _select_folder(self):
        self.grab_release()
        folder = filedialog.askdirectory(title="Select chosen-side concept folder")
        try:
            self.grab_set()
        except Exception:
            self.after(100, self._try_grab)
        self.focus_set()
        if folder:
            self.concept_path = folder
            self.concept_path_var.set(folder)

    def _try_grab(self):
        try:
            self.grab_set()
            self.focus_set()
        except Exception:
            pass

    def _run_analysis(self):
        if not self.concept_path:
            messagebox.showwarning("No Folder", "Pick a concept folder first.", parent=self)
            return
        try:
            bs = int(str(self.batch_size_var.get()).strip())
            if bs < 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid Batch Size", "Batch size must be a positive integer.", parent=self)
            return
        try:
            target = int(self.target_var.get())
        except ValueError:
            messagebox.showwarning("Invalid Target", "Invalid target resolution.", parent=self)
            return
        try:
            q = int(self.quant_var.get())
        except ValueError:
            messagebox.showwarning("Invalid Quantization", "Quantization must be an integer.", parent=self)
            return

        if self._analysis_thread and self._analysis_thread.is_alive():
            return

        self.status_var.set("Analyzing...")
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        ctk.CTkLabel(self.results_frame, text="Scanning images...", font=("", 14)).pack(pady=30)

        self._analysis_thread = threading.Thread(
            target=self._analyze_worker, args=(self.concept_path, bs, [target], q), daemon=True
        )
        self._analysis_thread.start()

    def _analyze_worker(self, concept_path, bs, targets, q):
        try:
            result = analyze_concept(concept_path, bs, targets, q)
            self.after(0, lambda: self._render_result(result))
        except Exception as ex:
            err = str(ex)
            self.after(0, lambda: self._render_error(err))

    def _render_error(self, message: str):
        self.status_var.set("")
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        ctk.CTkLabel(self.results_frame, text=f"Analysis failed: {message}", text_color="red", wraplength=900).pack(
            pady=20
        )

    def _render_result(self, result: dict):
        self._result = result
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        bs = result.get("batch_size", 1)
        q = result.get("quantization", 0)
        scanned = result.get("scanned", 0)
        unreadable = result.get("unreadable", 0)

        summary = (
            f"Scanned {scanned} image(s)"
            + (f", {unreadable} unreadable" if unreadable else "")
            + f"  -  batch_size={bs}, quantization={q}"
        )
        ctk.CTkLabel(self.results_frame, text=summary, font=("", 13), anchor="w").pack(fill="x", padx=5, pady=(0, 10))

        for target_data in result.get("targets", []):
            self._render_target_block(target_data, bs)

        if result.get("targets"):
            first = result["targets"][0]
            n_buckets = len(first["buckets"])
            total_pairs = first["total_pairs"]
            total_add = first["total_add"]
            total_remove = first["total_remove"]
            tgt = first["target"]
            self.status_var.set(
                f"{total_pairs} pairs across {n_buckets} buckets - "
                f"add {total_add} OR remove {total_remove} "
                f"for bs={bs} clean (target {tgt})."
            )
        else:
            self.status_var.set("No target resolutions analyzed.")

    def _render_target_block(self, target_data: dict, bs: int):
        card = ctk.CTkFrame(self.results_frame, border_width=1, border_color="gray30", corner_radius=8)
        card.pack(fill="x", padx=5, pady=(0, 12))

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 4))
        tgt = target_data["target"]
        ctk.CTkLabel(header, text=f"Target resolution: {tgt}", font=("", 14, "bold")).pack(side="left")
        tp = target_data["total_pairs"]
        td = target_data["total_drops"]
        ta = target_data["total_add"]
        tr = target_data["total_remove"]
        totals = f"Pairs {tp}   Drops {td}   Add {ta}   Remove {tr}"
        ctk.CTkLabel(header, text=totals, font=("", 12), text_color="gray").pack(side="right")

        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(fill="x", padx=12, pady=(4, 12))

        columns = ["Aspect", "H x W", "Count", "Drops", "Add for 0-drop", "Remove for 0-drop"]
        weights = [3, 2, 1, 1, 2, 2]
        for col_idx, (title, weight) in enumerate(zip(columns, weights, strict=True)):
            grid.grid_columnconfigure(col_idx, weight=weight)
            ctk.CTkLabel(grid, text=title, font=("", 12, "bold"), anchor="w").grid(
                row=0, column=col_idx, sticky="we", padx=6, pady=(0, 4)
            )

        for i, bucket in enumerate(target_data["buckets"], start=1):
            bh = bucket["h"]
            bw = bucket["w"]
            cells = [
                bucket["aspect_label"],
                f"{bh} x {bw}",
                str(bucket["count"]),
                str(bucket["drops"]),
                str(bucket["add"]),
                str(bucket["remove"]),
            ]
            fg = "red" if bucket["drops"] else None
            for col_idx, text in enumerate(cells):
                label = ctk.CTkLabel(grid, text=text, font=("", 12), anchor="w")
                if fg and col_idx == 3:
                    label.configure(text_color=fg)
                label.grid(row=i, column=col_idx, sticky="we", padx=6, pady=2)

        footer_row = len(target_data["buckets"]) + 1
        footer = [
            "TOTAL",
            "",
            str(target_data["total_pairs"]),
            str(target_data["total_drops"]),
            str(target_data["total_add"]),
            str(target_data["total_remove"]),
        ]
        for col_idx, text in enumerate(footer):
            ctk.CTkLabel(grid, text=text, font=("", 12, "bold"), anchor="w").grid(
                row=footer_row, column=col_idx, sticky="we", padx=6, pady=(6, 0)
            )
