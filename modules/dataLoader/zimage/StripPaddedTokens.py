from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class StripPaddedTokens(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_name: str,
            tokens_mask_name: str,
            hidden_state_name: str,
    ):
        super().__init__()
        self.tokens_name = tokens_name
        self.tokens_mask_name = tokens_mask_name
        self.hidden_state_name = hidden_state_name

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

        bool_mask = tokens_mask.bool()

        tokens = tokens[bool_mask]
        tokens_mask = tokens_mask[bool_mask]
        hidden_state = hidden_state[bool_mask]

        return {
            self.tokens_name: tokens,
            self.tokens_mask_name: tokens_mask,
            self.hidden_state_name: hidden_state,
        }
