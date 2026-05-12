from torch.utils.tensorboard import SummaryWriter
from modules.logger.base_logger import BaseLogger

class TensorboardLogger(BaseLogger):
    def __init__(self, run_dir, prefix=None):
        self.writer = SummaryWriter(log_dir=run_dir)
        self.prefix = prefix

    def log_scalar(self, tag, value, step):
        if self.prefix:
            tag = f'{self.prefix}/{tag}'
        self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, image, step):
        if self.prefix:
            tag = f'{self.prefix}/{tag}'
        self.writer.add_image(tag, image, step)

    def log_histogram(self, tag, values, step):
        if self.prefix:
            tag = f'{self.prefix}/{tag}'
        self.writer.add_histogram(tag, values, step)

    def log_text(self, tag, text, step):
        if self.prefix:
            tag = f'{self.prefix}/{tag}'
        self.writer.add_text(tag, text, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
