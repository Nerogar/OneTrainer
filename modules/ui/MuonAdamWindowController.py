from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.Optimizer import Optimizer
from modules.util.optimizer_util import OPTIMIZER_DEFAULT_PARAMETERS

MUON_AUX_ADAM_DEFAULTS = {
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.0,
}




class MuonAdamWindowController:
    def __init__(self, config: TrainConfig, parent_optimizer_type: Optimizer):
        self.config = config
        self.parent_optimizer_type = parent_optimizer_type

    def get_title(self) -> str:
        if self.parent_optimizer_type == Optimizer.MUON:
            return "Muon's Auxiliary AdamW Settings"
        return "Muon_adv's Auxiliary AdamW_adv Settings"

    def get_adam_params_def(self) -> dict:
        if self.parent_optimizer_type == Optimizer.MUON:
            return MUON_AUX_ADAM_DEFAULTS
        return OPTIMIZER_DEFAULT_PARAMETERS[Optimizer.ADAMW_ADV]
