from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch.nn.functional as F


class PadTokens(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_name: str,
            tokens_mask_name: str,
            hidden_state_name: str,
            max_length: int,
    ):
        super().__init__()
        self.tokens_name = tokens_name
        self.tokens_mask_name = tokens_mask_name
        self.hidden_state_name = hidden_state_name
        self.max_length = max_length

    def length(self) -> int:
        return self._get_previous_length(self.tokens_name)

    def get_inputs(self) -> list[str]:
        return [self.tokens_name, self.tokens_mask_name, self.hidden_state_name]

    def get_outputs(self) -> list[str]:
        return [self.tokens_name, self.tokens_mask_name, self.hidden_state_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.tokens_name, index)
        tokens_mask = self._get_previous_item(variation, self.tokens_mask_name, index)
        hidden_state = self._get_previous_item(variation, self.hidden_state_name, index)

        current_length = tokens.shape[0]

        if current_length >= self.max_length:
            return {
                self.tokens_name: tokens,
                self.tokens_mask_name: tokens_mask,
                self.hidden_state_name: hidden_state,
            }

        pad_length = self.max_length - current_length

        tokens = F.pad(tokens, (0, pad_length), value=0)
        tokens_mask = F.pad(tokens_mask, (0, pad_length), value=0)
        # hidden_state shape: [N, hidden_dim] -> [max_length, hidden_dim]
        hidden_state = F.pad(hidden_state, (0, 0, 0, pad_length), value=0.0)

        return {
            self.tokens_name: tokens,
            self.tokens_mask_name: tokens_mask,
            self.hidden_state_name: hidden_state,
        }
