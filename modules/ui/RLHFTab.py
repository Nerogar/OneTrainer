import json
import os
from tkinter import messagebox

from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_curation_util import check_dpo_pairs, dpo_concept_pairs
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DPOExecutionMode import DPOExecutionMode
from modules.util.enum.DPOPatienceMode import DPOPatienceMode
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
        components.label(self.scroll_frame, 0, 0, "RLHF Mode",
                         tooltip="Preference training method. DPO is the current option.")
        components.options_kv(self.scroll_frame, 0, 1, mode_options, self.ui_state, "rlhf_mode")

        components.label(self.scroll_frame, 1, 0, "Beta",
                         tooltip="How strongly the model follows your preferences. Higher values make changes more conservative, "
                                 "lower values make them more aggressive. Start with the default unless you have a reason to change it.")
        components.entry(self.scroll_frame, 1, 1, self.ui_state, "rlhf_dpo_beta")

        components.label(self.scroll_frame, 1, 3, "Enable RLHF",
                         tooltip="Turns on preference training. Use this with LoRA training and chosen/rejected concept pairs.")
        components.switch(self.scroll_frame, 1, 4, self.ui_state, "rlhf_enabled")

        components.label(self.scroll_frame, 2, 0, "Label Smoothing",
                         tooltip="Helps when your choices were a bit uncertain. Increase it slightly if you were not fully confident in some of the chosen/rejected picks.")
        components.entry(self.scroll_frame, 2, 1, self.ui_state, "rlhf_dpo_label_smoothing")

        components.label(self.scroll_frame, 2, 3, "DPO Validation",
                         tooltip="Checks DPO on a held-out set of chosen/rejected pairs so you can see whether preference training is generalizing.")
        components.switch(self.scroll_frame, 2, 4, self.ui_state, "rlhf_dpo_validation")

        components.label(self.scroll_frame, 3, 0, "Supervised Mix",
                         tooltip="Blends regular training with preference training. Higher values keep the model closer to what it already learned.")
        components.entry(self.scroll_frame, 3, 1, self.ui_state, "rlhf_supervised_mix")

        components.label(self.scroll_frame, 3, 3, "Validation %",
                         tooltip="How many prompt groups the DPO Pair Tool leaves out for validation.")
        components.entry(self.scroll_frame, 3, 4, self.ui_state, "rlhf_dpo_validation_percentage")

        components.label(self.scroll_frame, 4, 0, "Shared Noise",
                         tooltip="Uses the same noise for chosen and rejected comparisons, which makes the comparison fairer. Leave it on unless you're experimenting.")
        components.switch(self.scroll_frame, 4, 1, self.ui_state, "rlhf_dpo_shared_noise")

        execution_options = [
            ("Sequential", DPOExecutionMode.SEQUENTIAL),
            ("Policy Concurrent", DPOExecutionMode.POLICY_CONCURRENT),
            ("Full Concurrent", DPOExecutionMode.FULL_CONCURRENT),
        ]
        components.label(self.scroll_frame, 5, 0, "Execution Mode",
                         tooltip="Controls the VRAM and speed trade-off for DPO. Full Concurrent is the fastest and uses the most VRAM. "
                                 "Policy Concurrent is the middle ground for both speed and VRAM use. Sequential is the slowest and uses the least VRAM. "
                                 "Most users should leave this on Sequential.")
        components.options_kv(self.scroll_frame, 5, 1, execution_options, self.ui_state, "rlhf_dpo_execution_mode")

        components.label(self.scroll_frame, 4, 3, "Early Stopping",
                         tooltip="Stops DPO training when the validation signal stops improving.")
        components.switch(self.scroll_frame, 4, 4, self.ui_state, "rlhf_dpo_patience_enabled")

        components.label(self.scroll_frame, 5, 3, "Patience",
                         tooltip="How many validation checks can pass without improvement before training stops.")
        components.entry(self.scroll_frame, 5, 4, self.ui_state, "rlhf_dpo_patience_value")

        patience_mode_options = [
            ("Either", DPOPatienceMode.EITHER),
            ("Both", DPOPatienceMode.BOTH),
        ]
        components.label(self.scroll_frame, 6, 3, "Patience Mode",
                         tooltip="Either: stop when accuracy or chosen reward stalls. Both: stop only when both stall.")
        components.options_kv(self.scroll_frame, 6, 4, patience_mode_options, self.ui_state, "rlhf_dpo_patience_mode")

        components.button(self.scroll_frame, 7, 0, "Check Pairs", command=self._check_pairs,
                          tooltip="Check that your chosen and rejected concept folders line up before training.")

        components.label(self.scroll_frame, 8, 0, "Training Type:",
                         tooltip="Shows whether DPO is starting from a fresh adapter or refining a loaded adapter. The output is always an adapter file.")
        components.label(self.scroll_frame, 8, 1, training_type,
                         tooltip="DPO always writes an adapter file. New Adapter means the adapter starts from scratch. Existing Adapter means DPO refines a loaded adapter.")

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

        if result['format_stats']:
            fmt_parts = [f"{ext}: {count}" for ext, count in sorted(result['format_stats'].items())]
            lines.append(f"\nFormats: {', '.join(fmt_parts)}")

        for pair_info in result['pairs']:
            lines.append("\n--- Pair ---")
            chosen_path = pair_info.get("chosen_path")
            rejected_path = pair_info.get("rejected_path")
            chosen_stray = pair_info.get("chosen_stray", 0)
            rejected_stray = pair_info.get("rejected_stray", 0)
            lines.append(f"  Chosen: {chosen_path}")
            lines.append(f"  Rejected: {rejected_path}")
            lines.append(f"  Matched: {pair_info['matched']}, "
                         f"Chosen stray: {chosen_stray}, "
                         f"Rejected stray: {rejected_stray}")

        messagebox.showinfo("Check Pairs Results", "\n".join(lines))

    def _load_concept_pairs(self):
        concepts = self.train_config.concepts
        if concepts is None:
            concept_file = self.train_config.concept_file_name
            if not concept_file or not os.path.isfile(concept_file):
                raise RuntimeError("No concepts configured. Set up concepts in the Concepts tab first.")
            from modules.util.config.ConceptConfig import ConceptConfig
            with open(concept_file, 'r') as f:
                concepts = [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]
        concept_types = {ConceptType(concept.type) for concept in concepts if concept.enabled}
        if ConceptType.DPO_CHOSEN in concept_types or ConceptType.DPO_REJECTED in concept_types:
            return dpo_concept_pairs(concepts, is_validation=False)
        if ConceptType.DPO_CHOSEN_VAL in concept_types or ConceptType.DPO_REJECTED_VAL in concept_types:
            return dpo_concept_pairs(concepts, is_validation=True)
        raise RuntimeError("Need explicit chosen/rejected DPO concepts for training or validation.")
