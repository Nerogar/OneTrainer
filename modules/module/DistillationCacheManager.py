import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class DistillationCacheManager:
    """
    Manages caching of parent model predictions for distillation training.
    
    This allows a two-step distillation workflow:
    1. Generate cache: Run parent model inference and save predictions to disk
    2. Use cache: Load cached predictions instead of running parent model
    
    Benefits:
    - Reduces VRAM usage during training (no need to load parent model)
    - Faster training when using same dataset for multiple epochs
    - Avoids model swapping overhead on low-VRAM systems
    """
    
    def __init__(
        self,
        cache_dir: str,
        parent_model_path: str,
        parent_model_type: str,
        target_mode: str,
        cfg_scale: float,
        rollout_steps: int,
        rollout_blend: float,
    ):
        """
        Initialize the distillation cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            parent_model_path: Path to parent model (for validation)
            parent_model_type: Type of parent model (for validation)
        """
        self.cache_dir = Path(cache_dir)
        self.parent_model_path = parent_model_path
        self.parent_model_type = parent_model_type
        self.target_mode = target_mode
        self.cfg_scale = cfg_scale
        self.rollout_steps = rollout_steps
        self.rollout_blend = rollout_blend
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DistillationCacheManager] Initialized cache directory: {self.cache_dir.absolute()}")
        print(f"[DistillationCacheManager] Directory exists: {self.cache_dir.exists()}")
        print(f"[DistillationCacheManager] Directory writable: {os.access(self.cache_dir, os.W_OK)}")
        
        # Metadata file for cache validation
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _generate_cache_key(self, image_path: str, timestep: int) -> str:
        """
        Generate a unique cache key from image path and timestep.
        
        Uses hash to avoid filesystem path issues (special chars, length limits).
        
        Args:
            image_path: Path to the training image
            timestep: The timestep used for this prediction
            
        Returns:
            Hash-based cache key
        """
        # Combine image path and timestep for uniqueness
        key_string = f"{image_path}_{timestep}"
        # Use SHA256 hash (first 16 chars should be sufficient for uniqueness)
        hash_object = hashlib.sha256(key_string.encode())
        return hash_object.hexdigest()[:16]
    
    def _get_cache_filepath(self, cache_key: str) -> Path:
        """
        Get the full filepath for a cache entry.
        
        Args:
            cache_key: The cache key
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.pt"
    
    def save_prediction(
        self,
        image_path: str,
        timestep: int,
        prediction_dict: dict[str, Tensor],
        global_step: int = 0,
    ) -> None:
        """
        Save a parent model prediction to cache.
        
        Args:
            image_path: Path to the training image
            timestep: The timestep used for this prediction
            prediction_dict: Dictionary containing prediction tensors
                            Should include: 'predicted', 'target', 'prediction_type'
            global_step: Current training step (for debugging)
        """
        cache_key = self._generate_cache_key(image_path, timestep)
        cache_file = self._get_cache_filepath(cache_key)
        
        # Prepare cache data
        cache_data = {
            'predicted': prediction_dict.get('predicted').cpu() if prediction_dict.get('predicted') is not None else None,
            'target': prediction_dict.get('target').cpu() if prediction_dict.get('target') is not None else None,
            'predicted_empty': prediction_dict.get('predicted_empty').cpu() if prediction_dict.get('predicted_empty') is not None else None,
            'prediction_type': prediction_dict.get('prediction_type'),
            'timestep': timestep,
            'image_path': image_path,
            'global_step': global_step,
            'parent_model_path': self.parent_model_path,
            'parent_model_type': self.parent_model_type,
            'target_mode': self.target_mode,
            'cfg_scale': self.cfg_scale,
            'rollout_steps': self.rollout_steps,
            'rollout_blend': self.rollout_blend,
        }
        
        try:
            # Save to disk
            torch.save(cache_data, cache_file)
        except Exception as e:
            print(f"[DistillationCacheManager] ERROR saving cache file {cache_file}: {str(e)}")
            raise
    
    def load_prediction(self, image_path: str, timestep: int) -> dict[str, Any] | None:
        """
        Load a cached parent model prediction.
        
        Args:
            image_path: Path to the training image
            timestep: The timestep to load
            
        Returns:
            Dictionary with cached prediction data, or None if not found
        """
        cache_key = self._generate_cache_key(image_path, timestep)
        cache_file = self._get_cache_filepath(cache_key)
        
        if not cache_file.exists():
            self.cache_misses += 1
            return None
        
        try:
            cache_data = torch.load(cache_file, map_location='cpu')
            
            # Validate metadata
            if cache_data.get('parent_model_path') != self.parent_model_path:
                raise ValueError(
                    f"Cache metadata mismatch: parent_model_path\n"
                    f"  Expected: {self.parent_model_path}\n"
                    f"  Got: {cache_data.get('parent_model_path')}"
                )
            
            if cache_data.get('parent_model_type') != self.parent_model_type:
                raise ValueError(
                    f"Cache metadata mismatch: parent_model_type\n"
                    f"  Expected: {self.parent_model_type}\n"
                    f"  Got: {cache_data.get('parent_model_type')}"
                )

            if cache_data.get('target_mode') != self.target_mode:
                raise ValueError(
                    f"Cache metadata mismatch: target_mode\n"
                    f"  Expected: {self.target_mode}\n"
                    f"  Got: {cache_data.get('target_mode')}"
                )

            if float(cache_data.get('cfg_scale', 1.0)) != float(self.cfg_scale):
                raise ValueError(
                    f"Cache metadata mismatch: cfg_scale\n"
                    f"  Expected: {self.cfg_scale}\n"
                    f"  Got: {cache_data.get('cfg_scale')}"
                )

            if int(cache_data.get('rollout_steps', 1)) != int(self.rollout_steps):
                raise ValueError(
                    f"Cache metadata mismatch: rollout_steps\n"
                    f"  Expected: {self.rollout_steps}\n"
                    f"  Got: {cache_data.get('rollout_steps')}"
                )

            if float(cache_data.get('rollout_blend', 0.5)) != float(self.rollout_blend):
                raise ValueError(
                    f"Cache metadata mismatch: rollout_blend\n"
                    f"  Expected: {self.rollout_blend}\n"
                    f"  Got: {cache_data.get('rollout_blend')}"
                )
            
            # Validate predicted_empty for CFG_DISTILL mode
            if self.target_mode == 'CFG_DISTILL':
                if cache_data.get('predicted_empty') is None:
                    raise ValueError(
                        f"Cache data for CFG_DISTILL mode missing predicted_empty.\n"
                        f"This cache was likely generated with a different target_mode.\n"
                        f"Please regenerate the cache with target_mode='CFG_DISTILL'."
                    )
            
            self.cache_hits += 1
            return cache_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load cache file {cache_file}: {str(e)}")
    
    def cache_exists(self, image_path: str, timestep: int) -> bool:
        """
        Check if a cache entry exists for the given image and timestep.
        
        Args:
            image_path: Path to the training image
            timestep: The timestep
            
        Returns:
            True if cache exists
        """
        cache_key = self._generate_cache_key(image_path, timestep)
        cache_file = self._get_cache_filepath(cache_key)
        return cache_file.exists()
    
    def get_cache_stats(self) -> dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache_hits and cache_misses
        """
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
        }
    
    def clear_cache(self) -> int:
        """
        Delete all cache files in the cache directory.
        
        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pt"):
            cache_file.unlink()
            count += 1
        
        # Also remove metadata file if it exists
        if self.metadata_file.exists():
            self.metadata_file.unlink()
            count += 1
        
        return count
    
    def save_metadata(self, metadata: dict[str, Any]) -> None:
        """
        Save cache metadata to disk.
        
        Args:
            metadata: Metadata dictionary to save
        """
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self) -> dict[str, Any] | None:
        """
        Load cache metadata from disk.
        
        Returns:
            Metadata dictionary or None if file doesn't exist
        """
        if not self.metadata_file.exists():
            return None
        
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
