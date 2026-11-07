import copy
import json
import os
from pathlib import Path

from modules.util.training_queue import export_queue, load_base, parse_values


class QueueTrainingUIController:
    def __init__(self, train_config):
        self.train_config = train_config
        self.settings = {
            "mode": "Loop",
            "source": "Current UI",
            "base_path": "",
            "output_dir": "workspace/queue",
            "script_path": "training_queue.bat" if os.name == "nt" else "training_queue.sh",
            "power_action": "none",
        }
        self.base_snapshot = None
        self.rows = {mode: [] for mode in ("Loop", "Sweep")}

    def create_window(self, parent, view_cls):
        return view_cls(parent, self)

    def add_row(self, mode, **values):
        row = {"path": ""} if mode == "Loop" else {"key": "", "values": ""}
        row.update(values)
        self.rows[mode].append(row)
        return row

    def load(self, path):
        path = Path(path).resolve()
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec.get("version") != 1 or spec.get("mode") not in ("Loop", "Sweep"):
            raise ValueError("Choose saved Loop or Sweep settings.")
        self.base_snapshot = copy.deepcopy(spec.get("base_config"))
        self.settings.update(
            mode=spec["mode"],
            source="Saved snapshot" if self.base_snapshot is not None else "Current UI",
            base_path=spec.get("base_config_path", ""),
            output_dir=spec["output_dir"],
            power_action=spec.get("power_action", "none"),
            script_path=str(path.with_suffix(".bat" if os.name == "nt" else ".sh")),
        )
        self.rows["Loop"][:] = [{"path": source} for source in spec.get("configs", [])]
        self.rows["Sweep"][:] = [
            {"key": param["key"], "values": json.dumps(param["values"])}
            for param in spec.get("parameters", [])
        ]

    def make_settings(self):
        settings = self.settings
        spec = {
            "version": 1, "mode": settings["mode"],
            "output_dir": str(Path(settings["output_dir"]).resolve()),
            "power_action": settings["power_action"],
        }
        if settings["mode"] == "Loop":
            spec["configs"] = [str(Path(row["path"]).resolve()) for row in self.rows["Loop"] if row["path"].strip()]
            return spec
        if settings["source"] == "Saved snapshot":
            if self.base_snapshot is None:
                raise ValueError("Open a saved Sweep queue to use its snapshot.")
            spec["base_config"] = copy.deepcopy(self.base_snapshot)
        elif settings["source"] == "Current UI":
            spec["base_config"] = load_base(current=self.train_config)
        else:
            spec["base_config"] = load_base(settings["base_path"])
        if settings["source"] != "Current UI" and settings["base_path"]:
            spec["base_config_path"] = str(Path(settings["base_path"]).resolve())
        spec["parameters"] = []
        for row in self.rows["Sweep"]:
            key = row["key"].strip()
            if not key:
                raise ValueError("Every parameter row needs a key.")
            spec["parameters"].append({"key": key, "values": parse_values(row["values"])})
        return spec

    def export(self, *, overwrite=False):
        if not self.settings["script_path"].strip() or (
            self.settings["mode"] == "Sweep" and not self.settings["output_dir"].strip()
        ):
            raise ValueError("Choose an output directory and script path.")
        return export_queue(self.make_settings(), self.settings["script_path"], overwrite=overwrite)
