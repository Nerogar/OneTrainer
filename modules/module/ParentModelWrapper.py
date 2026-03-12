import torch
from abc import ABCMeta

from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin
from modules.util.config.TrainConfig import DistillationConfig, QuantizationConfig
from modules.util.enum.DataType import DataType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.quantization_util import replace_linear_with_quantized_layers
from modules.util.torch_util import torch_gc


class ParentModelWrapper:
    """
    Wrapper for managing a parent model used in distillation training.
    Handles loading, quantization, and device management for memory-efficient distillation.
    
    Similar to EMAModuleWrapper pattern, but loads an external model instead of
    copying parameters. Supports CPU offloading and quantization to reduce VRAM usage.
    """
    
    def __init__(
            self,
            config: DistillationConfig,
            model_loader: BaseModelLoader,
            train_device: str = "cuda",
            temp_device: str = "cpu",
    ):
        """
        Initialize the parent model wrapper.
        
        Args:
            config: Distillation configuration
            model_loader: Model loader for the parent model type
            train_device: Device for training (GPU)
            temp_device: Device for storage (typically CPU)
        """
        self.config = config
        self.model_loader = model_loader
        self.train_device = torch.device(train_device)
        self.temp_device = torch.device(temp_device)
        
        self.parent_model: BaseModel | None = None
        self._is_loaded = False
        self._is_on_train_device = False
        
    def load_parent_model(self) -> BaseModel:
        """
        Lazy-load the parent model from the configured path.
        Applies quantization if enabled. Model is loaded to temp_device (CPU) by default.
        
        Returns:
            The loaded parent model
        """
        if self._is_loaded:
            return self.parent_model
            
        if not self.config.enabled or not self.config.parent_model_path:
            raise ValueError("Distillation not enabled or parent model path not set")
        
        # Create model names for loading
        model_names = ModelNames(
            base_model=self.config.parent_model_path,
        )
        
        # Determine weight dtypes - use float32 for parent to avoid precision issues
        # Parent model inference doesn't need fp16
        weight_dtypes = ModelWeightDtypes.from_single_dtype(DataType.FLOAT_32)
        
        # Create a default quantization config (no SVD quantization for parent model)
        # Parent model uses its own quantization via distillation config
        quantization_config = QuantizationConfig.default_values()
        
        # Load the model
        # Initially load to temp_device (CPU) to save VRAM
        self.parent_model = self.model_loader.load(
            model_type=self.config.parent_model_type,
            model_names=model_names,
            weight_dtypes=weight_dtypes,
            quantization=quantization_config,
        )
        
        # Move to temp device initially
        if hasattr(self.parent_model, 'to'):
            self.parent_model.to(self.temp_device)
        
        # Apply quantization if enabled
        if self.config.quantize_parent and self.config.parent_quantization_dtype != DataType.NONE:
            self._quantize_parent_model()
        
        self._is_loaded = True
        self._is_on_train_device = False
        
        return self.parent_model
    
    def _quantize_parent_model(self):
        """
        Apply quantization to the parent model to reduce memory footprint.
        Uses existing quantization infrastructure from OneTrainer.
        Iterates through all nn.Module attributes of the parent model.
        """
        if not hasattr(self.parent_model, '__dict__'):
            return
        
        quantized_modules = []
            
        # Step 1: Replace linear layers with quantized versions
        for attr_name in dir(self.parent_model):
            # Skip private/protected attributes and methods
            if attr_name.startswith('_'):
                continue
                
            try:
                attr = getattr(self.parent_model, attr_name)
                
                # Check if this attribute is a PyTorch module (like unet, text_encoder, etc.)
                if isinstance(attr, torch.nn.Module):
                    # Apply quantization to this module
                    # No layer filtering for parent model - quantize all linear layers
                    replace_linear_with_quantized_layers(
                        parent_module=attr,
                        dtype=self.config.parent_quantization_dtype,
                        keep_in_fp32_modules=None,
                        quantization=None,  # Don't use SVD quantization for parent
                        copy_parameters=True,  # Copy parameters to quantized layers
                    )
                    quantized_modules.append(attr)
            except (AttributeError, RuntimeError):
                # Some attributes might not be accessible or quantizable
                # Continue with other attributes
                pass
        
        # Step 2: Initialize quantized layers by calling .quantize()
        # This is critical for QuantizedModuleMixin types (LinearNf4, LinearFp8, etc.)
        # They need .quantize() called to initialize internal state (e.g., quant_state)
        for module in quantized_modules:
            for child_module in module.modules():
                if isinstance(child_module, QuantizedModuleMixin):
                    # Set compute dtype to float32 (parent model doesn't need mixed precision)
                    child_module.compute_dtype = DataType.FLOAT_32.torch_dtype()
                    # Initialize quantization state
                    child_module.quantize(device=self.temp_device)
    
    def to_device(self, device: torch.device):
        """
        Move the parent model to a specific device.
        Used for temporarily moving to GPU for inference, then back to CPU.
        
        Args:
            device: Target device
        """
        if not self._is_loaded:
            raise RuntimeError("Parent model not loaded. Call load_parent_model() first.")
        
        current_device = device
        
        # Only move if device is different
        if current_device == self.train_device and not self._is_on_train_device:
            self.parent_model.to(device)
            self._is_on_train_device = True
        elif current_device == self.temp_device and self._is_on_train_device:
            self.parent_model.to(device)
            self._is_on_train_device = False
            # Critical: Force VRAM cleanup after moving quantized model back to CPU
            # Quantized layers have additional state buffers that need explicit cleanup
            torch_gc()
    
    def unload(self):
        """
        Explicitly unload the parent model and free memory.
        Should be called when training is complete.
        """
        if self._is_loaded:
            del self.parent_model
            self.parent_model = None
            self._is_loaded = False
            self._is_on_train_device = False
            torch_gc()
    
    def is_loaded(self) -> bool:
        """Check if parent model is loaded."""
        return self._is_loaded
    
    def is_on_train_device(self) -> bool:
        """Check if parent model is currently on the training device (GPU)."""
        return self._is_on_train_device
