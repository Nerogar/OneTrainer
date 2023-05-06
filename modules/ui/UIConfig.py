import os.path


class UIConfig:
    def __init__(self):
        self.config_name = ""
        self.configs = [("", "")]

        self.load("training_presets")

    def load(self, dir_path):
        if os.path.isdir(dir_path):
            for path in os.listdir(dir_path):
                if path.endswith(".json"):
                    name = os.path.basename(path)
                    name = os.path.splitext(name)[0]
                    path = os.path.join(dir_path, path)
                    self.configs.append((name, path))
