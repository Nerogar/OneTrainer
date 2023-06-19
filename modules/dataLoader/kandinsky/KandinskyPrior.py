import torch
from diffusers import KandinskyPriorPipeline
from mgds.MGDS import PipelineModule


class KandinskyPrior(PipelineModule):
    def __init__(
            self,
            in_name: str,
            out_name: str,
            prior_pipeline: KandinskyPriorPipeline,
    ):
        super(KandinskyPrior, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.prior_pipeline = prior_pipeline

    def length(self) -> int:
        return self.get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        text = self.get_previous_item(self.in_name, index)

        rand = self._get_rand(index)
        generator = torch.Generator(device=self.pipeline.device)
        generator.manual_seed(rand.randint(0, 1 << 30))

        pipeline_output = self.prior_pipeline(
            prompt=text,
            num_inference_steps=10,
            generator=generator,
        )
        embedding = pipeline_output.image_embeds.to(self.pipeline.device).squeeze()

        return {
            self.out_name: embedding,
        }
