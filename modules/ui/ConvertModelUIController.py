import traceback
from uuid import uuid4

from modules.util import create
from modules.util.args.ConvertModelArgs import ConvertModelArgs
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import EmbeddingName, ModelNames
from modules.util.torch_util import torch_gc


class ConvertModelUIController:
    def __init__(self):
        self.convert_model_args = ConvertModelArgs.default_values()
        self.view = None

    def create_window(self, parent, view_cls):
        self.view = view_cls(parent, self)
        return self.view

    def convert_model(self):
        try:
            self.view.set_converting(True)
            model_loader = create.create_model_loader(
                model_type=self.convert_model_args.model_type,
                training_method=self.convert_model_args.training_method
            )
            model_saver = create.create_model_saver(
                model_type=self.convert_model_args.model_type,
                training_method=self.convert_model_args.training_method
            )

            print("Loading model " + self.convert_model_args.input_name)
            if self.convert_model_args.training_method in [TrainingMethod.FINE_TUNE]:
                model = model_loader.load(
                    model_type=self.convert_model_args.model_type,
                    model_names=ModelNames(
                        base_model=self.convert_model_args.input_name,
                    ),
                    weight_dtypes=self.convert_model_args.weight_dtypes(),
                    quantization=QuantizationConfig.default_values(),
                )
            elif self.convert_model_args.training_method in [TrainingMethod.LORA, TrainingMethod.EMBEDDING]:
                model = model_loader.load(
                    model_type=self.convert_model_args.model_type,
                    model_names=ModelNames(
                        base_model=None,
                        lora=self.convert_model_args.input_name,
                        embedding=EmbeddingName(str(uuid4()), self.convert_model_args.input_name),
                    ),
                    weight_dtypes=self.convert_model_args.weight_dtypes(),
                    quantization=QuantizationConfig.default_values(),
                )
            else:
                raise Exception("could not load model: " + self.convert_model_args.input_name)

            print("Saving model " + self.convert_model_args.output_model_destination)
            model_saver.save(
                model=model,
                model_type=self.convert_model_args.model_type,
                output_model_format=self.convert_model_args.output_model_format,
                output_model_destination=self.convert_model_args.output_model_destination,
                dtype=self.convert_model_args.output_dtype.torch_dtype(),
            )
            print("Model converted")
        except Exception:
            traceback.print_exc()

        torch_gc()
        self.view.set_converting(False)
