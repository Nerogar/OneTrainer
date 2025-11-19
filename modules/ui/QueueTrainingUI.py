import itertools
import json
import platform
from pathlib import Path
from tkinter import filedialog, messagebox

from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class QueueTrainingUI(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.is_windows = platform.system().lower().startswith("win")
        self.repo_root = Path(__file__).resolve().parents[2]

        self.mode_var = ctk.StringVar(value="Loop")
        self.secrets_path_var = ctk.StringVar()
        self.script_path_var = ctk.StringVar(value="./queued_training/queue.bat")
        self.sweep_source_var = ctk.StringVar(value="current")
        self.base_config_path_var = ctk.StringVar()
        self.output_dir_var = ctk.StringVar(value="./queued_training/generated_configs")
        self.file_prefix_var = ctk.StringVar(value="queue_run")

        self.ui_state = UIState(self, {"script_path": "", "secrets_path": "", "base_config_path": ""})
        self.ui_state._UIState__vars.update({
            "script_path": self.script_path_var,
            "secrets_path": self.secrets_path_var,
            "base_config_path": self.base_config_path_var
        })

        self.loop_rows: list[dict] = []
        self.sweep_rows: list[dict] = []
        self.loop_row_values: list[str] = []
        self.sweep_row_values: list[tuple[str, str]] = []
        self.loop_row_counter = 0
        self.mode_frame: ctk.CTkFrame | None = None

        self.title("Training Queue Generator")
        self.geometry("820x640")
        self.minsize(720, 520)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_header_and_general()
        self._build_content_section()
        self._build_action_section()

        self.wait_visibility()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def _build_header_and_general(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 10))
        frame.grid_columnconfigure(1, weight=1)

        title = components.label(frame, row=0, column=0, text="Queue Training", pad=10,
                                font=ctk.CTkFont(size=18, weight="bold"))
        title.grid_configure(sticky="w")

        mode_selector = ctk.CTkSegmentedButton(frame, values=("Loop", "Sweep"),
                                              variable=self.mode_var, command=self._render_mode_frame)
        mode_selector.grid(row=0, column=1, padx=10, pady=10, sticky="e")
        mode_selector.set(self.mode_var.get())

        components.label(frame, row=1, column=0, text="Secrets file (optional)", pad=5)
        components.path_entry(frame, row=1, column=1, ui_state=self.ui_state, var_name="secrets_path",
                            is_output=False, path_type="file", allow_model_files=False,
                            valid_extensions=[".json"], width=80, sticky="ew")

        components.label(frame, row=2, column=0, text="Output script path", pad=5)
        components.path_entry(frame, row=2, column=1, ui_state=self.ui_state, var_name="script_path",
                            is_output=True, path_type="file", allow_model_files=False,
                            valid_extensions=[".bat"] if self.is_windows else [".sh"],
                            width=80, sticky="ew")

    def _build_content_section(self):
        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self._render_mode_frame()

    def _build_action_section(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        frame.grid_columnconfigure(0, weight=1)

        button_holder = ctk.CTkFrame(frame, fg_color="transparent")
        button_holder.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        generate_button = components.button(button_holder, row=0, column=0,
                                           text="Generate Script", command=self._generate_script)
        generate_button.grid_configure(sticky="e")

    def _choose_script_path(self):
        ext = ".bat" if self.is_windows else ".sh"
        if path := filedialog.asksaveasfilename(defaultextension=ext,
                filetypes=[(("Batch" if self.is_windows else "Shell"), f"*{ext}"), ("All Files", "*.*")]):
            self.script_path_var.set(path)

    def _render_mode_frame(self, *_args):
        self._snapshot_mode_state()
        if self.mode_frame:
            self.mode_frame.destroy()

        self.loop_rows = []
        self.sweep_rows = []

        self.mode_frame = ctk.CTkScrollableFrame(self.content_frame, fg_color="transparent")
        self.mode_frame.grid(row=0, column=0, sticky="nsew")

        (self._render_loop_mode if self.mode_var.get().lower() == "loop" else self._render_sweep_mode)()

    def _render_loop_mode(self):
        desc = components.label(self.mode_frame, row=0, column=0, pad=10, wraplength=760,
            text="Loop mode runs a fixed list of exported configs in sequence. "
                 "Each entry should point to a JSON config created via Export.")
        desc.configure(justify="left")
        desc.grid_configure(pady=(0, 10), sticky="w")

        container = ctk.CTkFrame(self.mode_frame, fg_color="transparent")
        container.grid(row=1, column=0, sticky="ew", padx=10)
        container.grid_columnconfigure(0, weight=1)
        self.loop_rows_container = container

        for value in (self.loop_row_values or [""]):
            self._add_loop_row(value)

        button_row = ctk.CTkFrame(self.mode_frame, fg_color="transparent")
        button_row.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        components.button(button_row, 0, 0, text="Add Config", command=self._add_loop_row).grid_configure(sticky="w")

    def _render_sweep_mode(self):
        desc = components.label(self.mode_frame, row=0, column=0, pad=10, wraplength=760,
            text="Sweep mode clones a base config (current UI or exported file) and varies specific keys. "
                 "Values are comma-separated and each combination generates a new config + queue entry. "
                 "\n You can also use range notation: start:end:step (e.g., '0.0001:0.001:0.0001' for learning rates).")
        desc.configure(justify="left")
        desc.grid_configure(pady=(0, 10), sticky="w")

        source_frame = ctk.CTkFrame(self.mode_frame, fg_color="transparent")
        source_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        source_frame.grid_columnconfigure(1, weight=1)

        components.label(source_frame, row=0, column=0, text="Base config source", pad=5).grid_configure(sticky="w")
        ctk.CTkRadioButton(source_frame, text="Use current UI settings", variable=self.sweep_source_var,
                          value="current", command=self._toggle_base_file_row).grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ctk.CTkRadioButton(source_frame, text="Use exported config file", variable=self.sweep_source_var,
                          value="file", command=self._toggle_base_file_row).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        self.base_file_row = ctk.CTkFrame(self.mode_frame, fg_color="transparent")
        self.base_file_row.grid(row=2, column=0, sticky="ew", padx=10)
        self.base_file_row.grid_columnconfigure(1, weight=1)
        components.label(self.base_file_row, row=0, column=0, text="Base config path", pad=5).grid_configure(sticky="w")
        components.path_entry(self.base_file_row, row=0, column=1, ui_state=self.ui_state, var_name="base_config_path",
                            is_output=False, path_type="file", allow_model_files=False,
                            valid_extensions=[".json"], width=80, sticky="ew")

        output_frame = ctk.CTkFrame(self.mode_frame, fg_color="transparent")
        output_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(10, 0))
        output_frame.grid_columnconfigure(1, weight=1)
        components.label(output_frame, row=0, column=0, text="Output directory for generated configs",
                        pad=5).grid_configure(sticky="w")
        ctk.CTkEntry(output_frame, textvariable=self.output_dir_var).grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        components.icon_button(output_frame, row=0, column=2, text="...",
                              command=self._choose_output_dir).grid_configure(padx=10, pady=5)

        components.label(output_frame, row=1, column=0, text="File prefix", pad=5).grid_configure(sticky="w")
        ctk.CTkEntry(output_frame, textvariable=self.file_prefix_var).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        components.label(self.mode_frame, row=4, column=0, pad=10,
            text="Sweep parameters (e.g., key: 'learning_rate', values: '1e-5, 2e-5, 3e-5' or '0.00001:0.00003:0.00001')"
        ).grid_configure(pady=(15, 5), sticky="w")

        container = ctk.CTkFrame(self.mode_frame, fg_color="transparent")
        container.grid(row=5, column=0, sticky="ew", padx=10)
        container.grid_columnconfigure(0, weight=1)
        self.sweep_params_container = container

        for key, values in (self.sweep_row_values or [("", "")]):
            self._add_sweep_param_row(key, values)

        sweep_button_row = ctk.CTkFrame(self.mode_frame, fg_color="transparent")
        sweep_button_row.grid(row=6, column=0, padx=10, pady=10, sticky="w")
        components.button(sweep_button_row, row=0, column=0, text="Add Parameter",
                         command=self._add_sweep_param_row).grid_configure(sticky="w")

        self._toggle_base_file_row()

    def _toggle_base_file_row(self):
        if self.sweep_source_var.get() == "file":
            self.base_file_row.grid()
        else:
            self.base_file_row.grid_remove()
            self.base_config_path_var.set("")

    def _choose_output_dir(self):
        if path := filedialog.askdirectory():
            self.output_dir_var.set(path)

    def _add_loop_row(self, initial_path: str = ""):
        row_frame = ctk.CTkFrame(self.loop_rows_container, fg_color="transparent")
        row_frame.grid_columnconfigure(1, weight=1)

        var_name = f"loop_config_{self.loop_row_counter}"
        self.loop_row_counter += 1

        var = ctk.StringVar(value=initial_path)
        self.ui_state._UIState__vars[var_name] = var

        components.label(row_frame, row=0, column=0, text=f"Config {len(self.loop_rows) + 1}",
                        pad=5, font=ctk.CTkFont(size=11)).grid_configure(sticky="w")

        components.path_entry(row_frame, row=0, column=1, ui_state=self.ui_state, var_name=var_name,
                            is_output=False, path_type="file", allow_model_files=False,
                            valid_extensions=[".json"], width=80, sticky="ew")

        components.button(row_frame, row=0, column=2, text="Remove",
                         command=lambda: self._remove_loop_row(row_frame), width=80).grid_configure(padx=10, pady=10)

        self.loop_rows.append({"frame": row_frame, "var": var})
        self._refresh_rows(self.loop_rows)

    def _remove_loop_row(self, frame):
        self.loop_rows = [row for row in self.loop_rows if row["frame"] != frame]
        frame.destroy()
        if not self.loop_rows:
            self._add_loop_row()
        else:
            self._refresh_rows(self.loop_rows)

    def _add_sweep_param_row(self, key: str = "", values: str = ""):
        row_frame = ctk.CTkFrame(self.sweep_params_container, fg_color="transparent")
        row_frame.grid_columnconfigure(0, weight=0)
        row_frame.grid_columnconfigure(1, weight=1)

        key_var = ctk.StringVar(value=key)
        value_var = ctk.StringVar(value=values)

        ctk.CTkEntry(row_frame, textvariable=key_var, width=200,
                    placeholder_text="learning_rate").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(row_frame, textvariable=value_var,
                    placeholder_text="1e-5, 2e-5, 3e-5").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        components.button(row_frame, row=0, column=2, text="Remove",
                         command=lambda: self._remove_sweep_row(row_frame), width=80).grid_configure(padx=5, pady=5)

        self.sweep_rows.append({"frame": row_frame, "key_var": key_var, "value_var": value_var})
        self._refresh_rows(self.sweep_rows)

    def _remove_sweep_row(self, frame):
        self.sweep_rows = [row for row in self.sweep_rows if row["frame"] != frame]
        frame.destroy()
        if not self.sweep_rows:
            self._add_sweep_param_row()
        else:
            self._refresh_rows(self.sweep_rows)

    def _refresh_rows(self, rows):
        for idx, row in enumerate(rows):
            row["frame"].grid(row=idx, column=0, sticky="ew")
            if "var" in row:
                label = row["frame"].winfo_children()[0]
                label.configure(text=f"Config {idx + 1}")

    def _snapshot_mode_state(self):
        if self.loop_rows:
            self.loop_row_values = [row["var"].get() for row in self.loop_rows]
        if self.sweep_rows:
            self.sweep_row_values = [(row["key_var"].get(), row["value_var"].get()) for row in self.sweep_rows]

    def _generate_script(self):
        if not (script_path := self.script_path_var.get().strip()):
            self._choose_script_path()
            if not (script_path := self.script_path_var.get().strip()):
                return

        config_paths = (self._collect_loop_configs() if self.mode_var.get().lower() == "loop"
                       else self._build_sweep_configs())

        if config_paths:
            commands = self._build_commands(config_paths)
            self._write_script(script_path, commands)
            messagebox.showinfo("Queue Training Generator", f"Created {len(commands)} queued runs at\n{script_path}")

    def _collect_loop_configs(self) -> list[str]:
        paths = []
        for row in self.loop_rows:
            if value := row["var"].get().strip():
                if not Path(value).is_file():
                    messagebox.showerror("Queue Training", f"Config file not found:\n{value}")
                    return []
                paths.append(str(Path(value).resolve()))

        if not paths:
            messagebox.showerror("Queue Training", "Add at least one config path.")
        return paths

    def _build_sweep_configs(self) -> list[str]:
        try:
            if not (base_dict := self._load_base_config()):
                return []
        except Exception as exc:
            messagebox.showerror("Queue Training", f"Failed to load base config:\n{exc}")
            return []

        if not (output_dir_str := self.output_dir_var.get().strip()):
            messagebox.showerror("Queue Training", "Select an output directory for generated configs.")
            return []
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.file_prefix_var.get().strip() or "queue_run"

        sweep_specs = []
        for row in self.sweep_rows:
            key = row["key_var"].get().strip()
            if not (values := self._parse_values(row["value_var"].get())) or not key:
                messagebox.showerror("Queue Training", "Each sweep row needs a key and at least one value.")
                return []
            sweep_specs.append((key, values))

        if not sweep_specs:
            messagebox.showerror("Queue Training", "Add at least one sweep parameter.")
            return []

        config_paths = []
        for index, combo in enumerate(itertools.product(*[values for _, values in sweep_specs]), start=1):
            config_copy = json.loads(json.dumps(base_dict))
            for (key, _), value in zip(sweep_specs, combo, strict=True):
                if not self._set_nested_value(config_copy, key, value):
                    messagebox.showerror("Queue Training", f"Invalid key path: {key}")
                    return []

            config_path = output_dir / f"{prefix}_{index:03d}.json"
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config_copy, handle, indent=4)
            config_paths.append(str(config_path.resolve()))

        return config_paths

    def _load_base_config(self) -> dict | None:
        if self.sweep_source_var.get() == "file":
            if not (path := self.base_config_path_var.get().strip()):
                messagebox.showerror("Queue Training", "Select a base config file.")
                return None
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return self.parent.train_config.to_settings_dict(secrets=False)

    def _parse_values(self, value_text: str) -> list:
        if value_text.count(':') == 2:
            try:
                parts = [float(x.strip()) for x in value_text.split(':')]
                start, end, step = parts
                if step == 0:
                    raise ValueError("Step cannot be zero")
                values = []
                current = start
                while (step > 0 and current <= end) or (step < 0 and current >= end):
                    values.append(current)
                    current += step
                return values
            except (ValueError, IndexError):
                pass

        return [self._parse_scalar(stripped) for chunk in value_text.split(",") if (stripped := chunk.strip())]

    def _parse_scalar(self, text: str):
        if (lowered := text.lower()) in {"true", "false"}:
            return lowered == "true"
        if lowered in {"none", "null"}:
            return None
        try:
            if text.startswith("0") and text not in {"0"} and not text.startswith("0."):
                raise ValueError
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            pass
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text.strip("\"'")

    def _set_nested_value(self, data: dict, key_path: str, value) -> bool:
        if not (tokens := self._split_key_path(key_path)):
            return False

        target = data
        for idx, (key, index) in enumerate(tokens):
            is_last = idx == len(tokens) - 1
            if key is not None:
                if not isinstance(target, dict) or key not in target:
                    return False
                if is_last:
                    target[key] = value
                    return True
                target = target[key]
            else:
                if not isinstance(target, list) or index is None or index >= len(target):
                    return False
                if is_last:
                    target[index] = value
                    return True
                target = target[index]
        return False

    def _split_key_path(self, key_path: str) -> list[tuple[str | None, int | None]]:
        tokens = []
        for raw_segment in key_path.split('.'):
            if not (segment := raw_segment.strip()):
                continue
            while segment:
                if '[' in segment:
                    before, _, remainder = segment.partition('[')
                    index_str, _, tail = remainder.partition(']')
                    if before:
                        tokens.append((before, None))
                    if not index_str.isdigit():
                        return []
                    tokens.append((None, int(index_str)))
                    segment = tail
                else:
                    tokens.append((segment, None))
                    segment = ""
        return tokens

    def _build_commands(self, config_paths: list[str]) -> list[str]:
        python_cmd = "python" if self.is_windows else "python3"
        train_script = ".\\scripts\\train.py" if self.is_windows else "./scripts/train.py"
        flags = f' --secrets-path "{self.secrets_path_var.get().strip()}"' if self.secrets_path_var.get().strip() else ""
        return [f'{python_cmd} {train_script} --config-path "{path}"{flags}' for path in config_paths]

    def _write_script(self, script_path: str, commands: list[str]):
        script_file = Path(script_path)
        script_file.parent.mkdir(parents=True, exist_ok=True)

        repo_root = self.repo_root.resolve()
        if self.is_windows:
            header = [
                "@echo off", "setlocal", f'set "ONE_TRAINER_DIR={repo_root}"',
                'if not exist "%ONE_TRAINER_DIR%" (',
                '    echo OneTrainer directory not found: "%ONE_TRAINER_DIR%"', "    exit /b 1", ")",
                'cd /d "%ONE_TRAINER_DIR%"', "call venv\\Scripts\\activate"
            ]
        else:
            header = [
                "#!/usr/bin/env bash", "set -e", f'ONE_TRAINER_DIR="{repo_root.as_posix()}"',
                'if [ ! -d "$ONE_TRAINER_DIR" ]; then',
                '  echo "OneTrainer directory not found: $ONE_TRAINER_DIR" >&2', "  exit 1", "fi",
                'cd "$ONE_TRAINER_DIR"', "source venv/bin/activate"
            ]

        with open(script_file, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(header + commands) + "\n")

        if not self.is_windows:
            script_file.chmod(script_file.stat().st_mode | 0o111)
