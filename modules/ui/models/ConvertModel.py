import traceback
from uuid import uuid4

from modules.ui.models.SingletonConfigModel import SingletonConfigModel
from modules.util import create
from modules.util.args.ConvertModelArgs import ConvertModelArgs
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import EmbeddingName, ModelNames
from modules.util.torch_util import torch_gc


class ConvertModel(SingletonConfigModel):
    def __init__(self):
        super().__init__(ConvertModelArgs.default_values())

    def convert_model(self):
        cfg = self.bulk_read("model_type", "training_method", "input_name",
                             "output_model_destination", "output_model_format", "output_dtype",
                             as_dict=True)

        try:
            model_loader = create.create_model_loader(
                model_type=cfg["model_type"],
                training_method=cfg["training_method"]
            )
            model_saver = create.create_model_saver(
                model_type=cfg["model_type"],
                training_method=cfg["training_method"]
            )

            print("Loading model " + cfg["input_name"])
            if cfg["training_method"] in [TrainingMethod.FINE_TUNE]:
                model = model_loader.load(
                    model_type=cfg["model_type"],
                    model_names=ModelNames(
                        base_model=cfg["input_name"],
                    ),
                    weight_dtypes=self.config.weight_dtypes(),
                )
            elif cfg["training_method"] in [TrainingMethod.LORA, TrainingMethod.EMBEDDING]:
                model = model_loader.load(
                    model_type=cfg["model_type"],
                    model_names=ModelNames(
                        lora=cfg["input_name"],
                        embedding=EmbeddingName(str(uuid4()), cfg["input_name"]),
                    ),
                    weight_dtypes=self.config.weight_dtypes(),
                )
            else:
                raise Exception("could not load model: " + cfg["input_name"])

            self.log("info", "Saving model " + cfg["output_model_destination"])
            model_saver.save(
                model=model,
                model_type=cfg["model_type"],
                output_model_format=cfg["output_model_format"],
                output_model_destination=cfg["output_model_destination"],
                dtype=cfg["output_dtype"].torch_dtype(),
            )
            self.log("info", "Model converted")
        except Exception:
            self.log("critical", traceback.format_exc())
        finally:
            torch_gc()
