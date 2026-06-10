import json
import os
from tkinter import messagebox

from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_curation_util import (
    check_dpo_pairs,
    correct_all_captions_to_chosen,
    find_caption_mismatches,
    fix_multiline_captions,
    remove_finalized_pair,
)
from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.RLHFMode import RLHFMode
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class RLHFTab:
    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__()

        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state

        self.scroll_frame = None

        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()
        self.scroll_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")

        self.scroll_frame.grid_columnconfigure(0, weight=0)
        self.scroll_frame.grid_columnconfigure(1, weight=1)
        self.scroll_frame.grid_columnconfigure(2, minsize=50)
        self.scroll_frame.grid_columnconfigure(3, weight=0)
        self.scroll_frame.grid_columnconfigure(4, weight=1)

        training_type = "DPO (Existing Adapter)" if self.train_config.lora_model_name else "DPO (New Adapter)"

        mode_options = [
            ("DPO", RLHFMode.DPO),
        ]
        components.label(
            self.scroll_frame, 0, 0, "RLHF Mode", tooltip="Preference training method. DPO is the current option."
        )
        components.options_kv(self.scroll_frame, 0, 1, mode_options, self.ui_state, "rlhf_mode")

        components.label(
            self.scroll_frame,
            1,
            0,
            "Beta",
            tooltip="How strongly the model follows your preferences. Higher values make changes more conservative, "
            "lower values make them more aggressive. Start with the default unless you have a reason to change it.",
        )
        components.entry(self.scroll_frame, 1, 1, self.ui_state, "rlhf_dpo_beta")

        components.label(
            self.scroll_frame,
            1,
            3,
            "Enable RLHF",
            tooltip="Turns on preference training. Use this with LoRA training and chosen/rejected concept pairs.",
        )
        components.switch(self.scroll_frame, 1, 4, self.ui_state, "rlhf_enabled")

        components.label(
            self.scroll_frame,
            2,
            0,
            "Label Smoothing",
            tooltip="Helps when your choices were a bit uncertain. Increase it slightly if you were not fully confident in some of the chosen/rejected picks.",
        )
        components.entry(self.scroll_frame, 2, 1, self.ui_state, "rlhf_dpo_label_smoothing")

        components.label(
            self.scroll_frame,
            2,
            3,
            "DPO Validation",
            tooltip="Checks DPO on a held-out set of chosen/rejected pairs so you can see whether preference training is generalizing.",
        )
        components.switch(self.scroll_frame, 2, 4, self.ui_state, "rlhf_dpo_validation")

        components.label(
            self.scroll_frame,
            3,
            0,
            "Supervised Mix",
            tooltip="Blends an NLL term on the CHOSEN images of each DPO pair into the loss. "
            "Anchors the policy to its own chosen samples.",
        )
        components.entry(self.scroll_frame, 3, 1, self.ui_state, "rlhf_supervised_mix")

        objective_options = [
            ("DPO (sigmoid)", DPOObjective.SIGMOID),
            ("IPO", DPOObjective.IPO),
        ]
        components.label(
            self.scroll_frame,
            4,
            0,
            "Objective",
            tooltip="DPO (sigmoid): the standard preference loss, scaled by Beta — pushes the preference margin "
            "ever higher, which can reward-hack on long runs. "
            "IPO: regresses the margin toward the fixed target 1/(2*Tau) instead — bounded by construction, "
            "more resistant to reward hacking. Beta and Label Smoothing do not apply to IPO.",
        )
        components.options_kv(self.scroll_frame, 4, 1, objective_options, self.ui_state, "rlhf_dpo_objective")

        components.label(
            self.scroll_frame,
            5,
            0,
            "IPO Tau",
            tooltip="IPO regularization strength; the target preference margin is 1/(2*Tau). Diffusion DPO margins "
            "are tiny (around 1e-3), so Tau in the hundreds to thousands gives realistic targets. "
            "Larger Tau = smaller target = gentler training.",
        )
        components.entry(self.scroll_frame, 5, 1, self.ui_state, "rlhf_dpo_ipo_tau")

        components.label(
            self.scroll_frame,
            6,
            0,
            "Adaptive Beta",
            tooltip="Adjusts Beta per optimizer step from the smoothed reward margin (beta-DPO): informative "
            "batches with above-average margins raise Beta, uninformative ones lower it, bounded to "
            "[Beta/4, Beta*4]. Beta is the starting point. Only applies to the DPO (sigmoid) objective. "
            "Note that Beta and learning rate are coupled - if you double Beta, halve the learning rate.",
        )
        components.switch(self.scroll_frame, 6, 1, self.ui_state, "rlhf_dpo_adaptive_beta")

        components.label(
            self.scroll_frame,
            3,
            3,
            "Validation %",
            tooltip="How many prompt groups the DPO Pair Tool leaves out for validation.",
        )
        components.entry(self.scroll_frame, 3, 4, self.ui_state, "rlhf_dpo_validation_percentage")

        components.label(
            self.scroll_frame,
            4,
            3,
            "Early Stopping",
            tooltip="Stops DPO training when the validation signal stops improving.",
        )
        components.switch(self.scroll_frame, 4, 4, self.ui_state, "rlhf_dpo_patience_enabled")

        components.label(
            self.scroll_frame,
            5,
            3,
            "Patience",
            tooltip="How many validation checks can pass without improvement before training stops.",
        )
        components.entry(self.scroll_frame, 5, 4, self.ui_state, "rlhf_dpo_patience_value")

        components.label(
            self.scroll_frame,
            6,
            3,
            "Save Best",
            tooltip="Saves a checkpoint when validation accuracy hits a new high. "
            "The best checkpoint is restored at the end of training.",
        )
        components.switch(self.scroll_frame, 6, 4, self.ui_state, "rlhf_dpo_save_best")

        components.label(
            self.scroll_frame,
            8,
            3,
            "Timestep Margins",
            tooltip="Logs the reward margin bucketed by timestep quartile (dpo/margin_by_t/q1..q4) to TensorBoard. "
            "Useful to verify pairs are compared evenly across noise levels.",
        )
        components.switch(self.scroll_frame, 8, 4, self.ui_state, "rlhf_dpo_timestep_margin_logging")

        components.button(
            self.scroll_frame,
            7,
            0,
            "Check Pairs",
            command=self._check_pairs,
            tooltip="Check that your chosen and rejected concept folders line up before training.",
        )
        components.button(
            self.scroll_frame,
            7,
            1,
            "Review Pairs",
            command=self._review_pairs,
            tooltip="Visually review your chosen/rejected image pairs. Remove bad pairs and their counterparts.",
        )
        components.button(
            self.scroll_frame,
            7,
            3,
            "DPO Bucket Analysis",
            command=self._bucket_analysis,
            tooltip="Show per-aspect-bucket pair counts and what to add or remove for clean batches at a given batch size.",
        )

        components.label(
            self.scroll_frame,
            8,
            0,
            "Training Type:",
            tooltip="Shows whether DPO is starting from a fresh adapter or refining a loaded adapter. The output is "
            "always an adapter file. DPO requires LoRA training; full finetuning is not supported.",
        )
        components.label(
            self.scroll_frame,
            8,
            1,
            training_type,
            tooltip="DPO always writes an adapter file. New Adapter means the adapter starts from scratch. Existing Adapter means DPO refines a loaded adapter.",
        )

    def _check_pairs(self):
        try:
            concept_pairs = self._load_concept_pairs()
        except Exception as ex:
            messagebox.showerror("Check Pairs Error", str(ex))
            return

        try:
            result = check_dpo_pairs(concept_pairs)
        except Exception as ex:
            messagebox.showerror("Check Pairs Error", f"Error checking pairs: {ex}")
            return

        total_matched = result.get("total_matched", 0)
        total_chosen_stray = result.get("total_chosen_stray", 0)
        total_rejected_stray = result.get("total_rejected_stray", 0)

        lines = [f"Matched pairs: {total_matched}"]
        if total_chosen_stray > 0:
            lines.append(f"Chosen strays (no rejected match): {total_chosen_stray}")
        if total_rejected_stray > 0:
            lines.append(f"Rejected strays (no chosen match): {total_rejected_stray}")

        if result["format_stats"]:
            fmt_parts = [f"{ext}: {count}" for ext, count in sorted(result["format_stats"].items())]
            lines.append(f"\nFormats: {', '.join(fmt_parts)}")

        for pair_info in result["pairs"]:
            lines.append("\n--- Pair ---")
            chosen_path = pair_info.get("chosen_path")
            rejected_path = pair_info.get("rejected_path")
            chosen_stray = pair_info.get("chosen_stray", 0)
            rejected_stray = pair_info.get("rejected_stray", 0)
            lines.append(f"  Chosen: {chosen_path}")
            lines.append(f"  Rejected: {rejected_path}")
            lines.append(
                f"  Matched: {pair_info['matched']}, Chosen stray: {chosen_stray}, Rejected stray: {rejected_stray}"
            )

        multiline = result.get("multiline_captions", 0)
        if multiline:
            lines.append(f"\nMultiline captions: {multiline}")

        has_strays = total_chosen_stray > 0 or total_rejected_stray > 0
        if has_strays:
            lines.append(f"\nTotal strays: {total_chosen_stray + total_rejected_stray}")

        messagebox.showinfo("Check Pairs Results", "\n".join(lines))

        if has_strays:
            remove = messagebox.askyesno(
                "Remove Strays?",
                f"Found {total_chosen_stray + total_rejected_stray} stray file(s) with no matching pair.\n\n"
                f"Remove them and their caption files?",
            )
            if remove:
                removed = self._remove_strays(concept_pairs, result)
                messagebox.showinfo("Strays Removed", f"Removed {removed} stray file(s) and their captions.")

        if multiline:
            fix = messagebox.askyesno(
                "Fix Multiline Captions?",
                f"Found {multiline} caption file(s) with newlines.\n\n"
                f"Newlines are treated as separate captions by OneTrainer, which can "
                f"sabotage training. Flatten them to single-line captions?",
            )
            if fix:
                fixed = fix_multiline_captions(concept_pairs)
                messagebox.showinfo("Captions Fixed", f"Flattened {fixed} caption file(s) to single lines.")

        # Caption-content mismatch resolution: chosen.txt vs rejected.txt for
        # matched image pairs. Runs last so any earlier stray/multiline fixes
        # are reflected in the matched-pair set this scans.
        try:
            mismatches = find_caption_mismatches(concept_pairs)
        except Exception as ex:
            messagebox.showerror("Caption Mismatch Error", f"Error checking captions: {ex}")
            return

        if not mismatches:
            return

        from modules.ui.DPOCaptionMismatchWindow import (
            DPOCaptionMismatchChoiceDialog,
            DPOCaptionMismatchWindow,
        )

        dialog = DPOCaptionMismatchChoiceDialog(self.master.winfo_toplevel(), len(mismatches))
        self.master.winfo_toplevel().wait_window(dialog)
        choice = dialog.result
        if choice == "correct_all":
            corrected = correct_all_captions_to_chosen(mismatches)
            messagebox.showinfo(
                "Captions Corrected",
                f"Overwrote {corrected} rejected caption file(s) with the chosen caption.",
            )
        elif choice == "manual":
            DPOCaptionMismatchWindow(self.master.winfo_toplevel(), concept_pairs)

    def _remove_strays(self, concept_pairs, result) -> int:
        from modules.util.dpo_curation_util import dpo_pair_key
        from modules.util.path_util import supported_image_extensions

        exts = supported_image_extensions()
        removed = 0

        for i, (chosen_path, rejected_path) in enumerate(concept_pairs):
            pair_info = result["pairs"][i]
            if pair_info.get("chosen_stray", 0) == 0 and pair_info.get("rejected_stray", 0) == 0:
                continue

            chosen_keys: dict[str, str] = {}
            rejected_keys: dict[str, str] = {}

            for root, _dirs, files in os.walk(chosen_path):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in exts:
                        full = os.path.join(root, fname)
                        chosen_keys[dpo_pair_key(full, chosen_path)] = full

            for root, _dirs, files in os.walk(rejected_path):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in exts:
                        full = os.path.join(root, fname)
                        rejected_keys[dpo_pair_key(full, rejected_path)] = full

            matched = set(chosen_keys) & set(rejected_keys)

            for key, path in chosen_keys.items():
                if key not in matched:
                    remove_finalized_pair(path, None)
                    removed += 1
            for key, path in rejected_keys.items():
                if key not in matched:
                    remove_finalized_pair(None, path)
                    removed += 1

        return removed

    def _review_pairs(self):
        try:
            concept_pairs = self._load_concept_pairs()
        except Exception as ex:
            messagebox.showerror("Review Pairs Error", str(ex))
            return

        from modules.ui.DPOReviewWindow import DPOReviewWindow

        DPOReviewWindow(self.master.winfo_toplevel(), concept_pairs)

    def _bucket_analysis(self):
        from modules.ui.DPOBucketAnalysisWindow import DPOBucketAnalysisWindow

        DPOBucketAnalysisWindow(self.master.winfo_toplevel(), self.train_config)

    def _load_concept_pairs(self):
        concept_pairs = dpo_concept_pattern_dirs(self._load_concepts())
        if not concept_pairs:
            raise RuntimeError(
                "No DPO concepts found. Set the chosen/rejected patterns on a concept in the Concepts tab."
            )
        return concept_pairs

    def _load_concepts(self):
        concepts = self.train_config.concepts
        if concepts is None:
            concept_file = self.train_config.concept_file_name
            if not concept_file or not os.path.isfile(concept_file):
                raise RuntimeError("No concepts configured. Set up concepts in the Concepts tab first.")
            from modules.util.config.ConceptConfig import ConceptConfig

            with open(concept_file, "r") as f:
                concepts = [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]
        return concepts
