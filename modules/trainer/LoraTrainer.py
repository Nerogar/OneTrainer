from modules.trainer.BaseTrainer import BaseTrainer
from modules.util.args.TrainArgs import TrainArgs


class LoraTrainer(BaseTrainer):
    def __init__(self, args: TrainArgs):
        super(LoraTrainer, self).__init__(args=args)

    def start(self):
        model_loader = self.create_model_loader()
        model_setup = self.create_model_setup()
