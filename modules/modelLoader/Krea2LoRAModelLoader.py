from modules.model.Krea2Model import Krea2Model
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.krea2.Krea2LoRALoader import Krea2LoRALoader
from modules.modelLoader.krea2.Krea2ModelLoader import Krea2ModelLoader
from modules.util.enum.ModelType import ModelType

Krea2LoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.KREA_2: "resources/sd_model_spec/krea2-lora.json"},
    model_class=Krea2Model,
    model_loader_class=Krea2ModelLoader,
    embedding_loader_class=None,
    lora_loader_class=Krea2LoRALoader,
)
