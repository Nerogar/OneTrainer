import copy
import json
import os

from modules.ui.models.SingletonConfigModel import SingletonConfigModel
from modules.ui.models.StateModel import StateModel
from modules.util import path_util
from modules.util.config.SampleConfig import SampleConfig
from modules.util.path_util import write_json_atomic


class SampleModel(SingletonConfigModel):
    def __init__(self):
        super().__init__([])

    def __len__(self):
        with self.critical_region_read():
            return len(self.config)

    def get_default_sample(self):
        return SampleConfig.default_values().to_dict()

    def create_new_sample(self):
        with self.critical_region_write():
            smp_cfg = SampleConfig.default_values()
            self.config.append(smp_cfg)

    def clone_sample(self, idx):
        with self.critical_region_write():
            new_element = copy.deepcopy(self.config[idx])
            self.config.append(new_element)

    def delete_sample(self, idx):
        with self.critical_region_write():
            self.config.pop(idx)

    def toggle_samples(self):
        some_enabled = self.some_samples_enabled()

        with self.critical_region_write():
            for smp in self.config:
                smp.enabled = not some_enabled

    def some_samples_enabled(self):
        with self.critical_region_read():
            out = False
            for smp in self.config:
                out |= smp.enabled
            return out

    def save_config(self, path="training_samples"):
        if not os.path.exists(path):
            os.mkdir(path)

        config_path = StateModel.instance().get_state("sample_definition_file_name")
        with self.critical_region_read():
            write_json_atomic(config_path, [element.to_dict() for element in self.config])

    def load_config(self, filename, path="training_samples"):
        if not os.path.exists(path):
            os.mkdir(path)

        if filename == "":
            filename = "samples"

        config_file = path_util.canonical_join(path, f"{filename}.json")
        StateModel.instance().set_state("sample_definition_file_name", config_file)

        with self.critical_region_write():
            self.config = []

            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    loaded_config_json = json.load(f)
                    for element_json in loaded_config_json:
                        element = SampleConfig.default_values().from_dict(element_json)
                        self.config.append(element)
