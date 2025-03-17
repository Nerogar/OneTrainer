import gc
import logging
import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path

from modules.module.BaseImageMaskModel import (
    BaseImageMaskModel,
    MaskSample,
)
from modules.module.captioning.sample import CaptionSample
from modules.module.Moondream2Model import Moondream2Model

import torch

import cv2
import numpy as np
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy import ndimage

# Create logger for this module
logger = logging.getLogger(__name__)


class ModelCache:
    def __init__(self):
        self.models = {}

    def get(self, model_type, use_cuda):
        """Get a model from cache or return None"""
        key = f"{model_type}_{use_cuda}"
        if key in self.models:
            logger.info(f"Using cached {model_type} model...")
            return self.models[key]
        return None

    def put(self, model_type, model, use_cuda):
        """Store a model in the cache"""
        key = f"{model_type}_{use_cuda}"
        self.models[key] = model
        return model

    def remove(self, model_type, use_cuda):
        """Remove a model from cache and free memory"""
        key = f"{model_type}_{use_cuda}"
        if key in self.models:
            logger.info(f"Unloading {model_type} model...")
            del self.models[key]
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()


MODEL_CACHE = ModelCache()


class ObjectDetector:
    """Handles object detection with Moondream model"""

    def __init__(self, device: torch.device, dtype: torch.dtype, model_revision: str = "05d640e6da70c37b2473e0db8fef0233c0709ce4"):
        self.device = device
        self.dtype = dtype
        self.use_cuda = device.type == "cuda"
        self.model_revision = model_revision
        self.model = None

    def load(self, cache=True, model_manager=None):
        """Load Moondream2 model with optional caching."""
        # First, check if the parent model manager already has a loaded captioning model
        if model_manager is not None:
            if (hasattr(model_manager, 'captioning_model') and
                model_manager.captioning_model is not None and
                hasattr(model_manager.captioning_model, '__class__') and
                model_manager.captioning_model.__class__.__name__ == 'Moondream2Model'):

                logger.info("Reusing existing Moondream2 captioning model from model manager")
                self.model = model_manager.captioning_model
                return self

        # Then check local cache
        if cache:
            cached_model = MODEL_CACHE.get("moondream", self.use_cuda)
            if cached_model is not None:
                self.model = cached_model
                return self

        logger.info("Loading Moondream model...")
        self.model = Moondream2Model(
            device=self.device,
            dtype=self.dtype,
            model_revision=self.model_revision,
            stream=False,
        )

        if cache:
            MODEL_CACHE.put("moondream", self.model, self.use_cuda)

        return self

    def unload(self):
        """Unload model and free memory"""
        if self.model:
            MODEL_CACHE.remove("moondream", self.use_cuda)
            self.model = None

    def detect(self, image_path: str, prompt: str, max_objects=5):
        """Detect objects in an image using Moondream2."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Create caption sample for Moondream2
        caption_sample = CaptionSample(image_path)

        # Load image
        image = Image.open(image_path)

        # Get image dimensions and numpy array
        img_width, img_height = image.size
        np_image = np.array(image)

        # Detect objects using Moondream2
        detection_results = self.model.generate_detection(caption_sample, prompt)

        if not detection_results or 'objects' not in detection_results:
            logger.info(f"No objects detected for prompt: '{prompt}'")
            return None, [], (0, 0)

        objects = detection_results["objects"][:max_objects]

        if not objects:
            logger.info(f"No objects found for prompt: '{prompt}'")
            return None, [], (0, 0)

        # Convert box format if needed
        for obj in objects:
            if 'box' in obj and len(obj['box']) == 4:
                x_min, y_min, x_max, y_max = obj['box']

                # Add normalized coordinates for compatibility with segmenter
                if all(0 <= coord <= 1 for coord in [x_min, y_min, x_max, y_max]):
                    obj["x_min"] = x_min
                    obj["y_min"] = y_min
                    obj["x_max"] = x_max
                    obj["y_max"] = y_max
                else:
                    # Convert to normalized if not already
                    obj["x_min"] = x_min / img_width
                    obj["y_min"] = y_min / img_height
                    obj["x_max"] = x_max / img_width
                    obj["y_max"] = y_max / img_height

        # Filter objects with valid coordinates
        filtered_objects = [
            obj for obj in objects
            if all(k in obj for k in ["x_min", "y_min", "x_max", "y_max"])
        ]

        if not filtered_objects:
            logger.info("No valid bounding boxes found.")
            return None, [], (0, 0)

        return np_image, filtered_objects, (img_width, img_height)


class Segmenter:
    """Handles segmentation with SAM2 model"""

    def __init__(self, device: torch.device, sam2_model_name: str):
        self.device = device
        self.use_cuda = device.type == "cuda"
        self.sam2_model_name = sam2_model_name
        self.predictor = None

    def load(self, cache=True):
        """Load SAM2 model with optional caching."""
        if cache:
            cached_model = MODEL_CACHE.get("sam", self.use_cuda)
            if cached_model is not None:
                self.predictor = cached_model
                return self

        logger.info(f"Loading SAM2 model '{self.sam2_model_name}'...")
        try:
            self.predictor = SAM2ImagePredictor.from_pretrained(self.sam2_model_name)
            if self.use_cuda:
                self.predictor.model.to(device=self.device)
                logger.info("SAM2 model loaded on CUDA")
        except Exception as e:
            logger.error(f"Error loading SAM2 model: {str(e)}")
            raise

        if cache:
            MODEL_CACHE.put("sam", self.predictor, self.use_cuda)

        return self

    def unload(self):
        """Unload model and free memory"""
        if self.predictor:
            MODEL_CACHE.remove("sam", self.use_cuda)
            self.predictor = None

    def segment(
        self,
        np_image,
        detected_objects,
        image_dimensions,
        threshold: float = 0.3,
        smooth_pixels: int = 5,
        expand_pixels: int = 10,
        min_region_size: int = 32,
    ):
        """Segment detected objects using SAM2."""
        if not self.predictor:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            self.predictor.set_image(np_image)
        except Exception as e:
            logger.error(f"Error setting image for SAM2: {str(e)}")
            return [], []

        if not detected_objects:
            return [], []

        # Unpack image dimensions
        img_width, img_height = image_dimensions

        # Pre-compute all coordinates and tensors at once
        boxes_px = []
        box_prompts = []
        point_prompts = []
        point_labels = []
        areas = []

        for obj in detected_objects:
            # Convert normalized coordinates directly to integers
            x_min_px = max(0, int(obj["x_min"] * img_width))
            y_min_px = max(0, int(obj["y_min"] * img_height))
            x_max_px = min(img_width, int(obj["x_max"] * img_width))
            y_max_px = min(img_height, int(obj["y_max"] * img_height))

            # Calculate area and padding in one step
            area = (x_max_px - x_min_px) * (y_max_px - y_min_px)
            padding = max(1, int(area * 0.00001 + 0.5))
            areas.append(area)

            # Apply padding to bounding box
            x_min_padded = max(0, x_min_px - padding)
            y_min_padded = max(0, y_min_px - padding)
            x_max_padded = min(img_width, x_max_px + padding)
            y_max_padded = min(img_height, y_max_px + padding)

            bbox = [x_min_padded, y_min_padded, x_max_padded, y_max_padded]
            boxes_px.append(bbox)
            box_prompts.append(bbox)

            # Calculate center point directly
            point_prompts.append(
                [
                    (x_min_padded + x_max_padded) // 2,
                    (y_min_padded + y_max_padded) // 2,
                ]
            )
            point_labels.append(1)

        # Create all tensors at once for better memory efficiency
        dtype = torch.float16 if self.use_cuda else torch.float32

        # Move tensors to device immediately to reduce memory transfers
        box_tensor = torch.tensor(box_prompts, dtype=dtype, device=self.device)
        point_tensor = torch.tensor(point_prompts, dtype=dtype, device=self.device)
        label_tensor = torch.tensor(point_labels, dtype=torch.int, device=self.device)

        # Pre-calculate adaptive min sizes for all objects
        adaptive_min_sizes = [max(min_region_size, int(area * 0.0001)) for area in areas]

        # Create reusable mask refinement function
        refine_partial = partial(
            MaskUtility.refine_mask,
            fill_holes=True,
            smooth=True,
            smooth_pixels=smooth_pixels,
            remove_small=True,
            min_size_ratio=0.01,
        )

        refined_masks = []
        num_objects = len(detected_objects)

        try:
            with torch.inference_mode():
                context_manager = torch.amp.autocast(self.device.type) if self.use_cuda else nullcontext()

                with context_manager:
                   # Process each object individually - this is the most reliable approach
                    for i in range(num_objects):
                        # Prepare tensors for this object
                        box_slice = box_tensor[i:i+1]
                        point_slice = point_tensor[i:i+1]
                        label_slice = label_tensor[i:i+1]

                        # Skip invalid tensors early without try-except
                        if box_slice.numel() == 0 or point_slice.numel() == 0:
                            logger.warning(f"Skipping object {i+1}/{num_objects}: Invalid input tensors")
                            continue

                        # Pre-declare variables outside try block
                        masks, scores, best_mask = None, None, None

                        try:
                            # Only put the actual prediction in try-except for better performance
                            masks, scores, _ = self.predictor.predict(
                                box=box_slice,
                                point_coords=point_slice,
                                point_labels=label_slice,
                                multimask_output=True,
                            )
                        except Exception as e:
                            logger.error(f"Error predicting mask for object {i+1}/{num_objects}: {e}")
                            continue

                        # Process results outside the try block when possible
                        best_idx = torch.argmax(scores).item() if torch.is_tensor(scores) else np.argmax(scores)
                        best_mask = masks[best_idx]

                        if torch.is_tensor(best_mask):
                            best_mask = best_mask.cpu().numpy()

                        # Refine mask
                        raw_mask = best_mask > threshold
                        refined_mask = refine_partial(
                            raw_mask,
                            area=areas[i],
                            min_size_abs=adaptive_min_sizes[i],
                            dilate_pixels=expand_pixels,
                            smooth_pixels=smooth_pixels
                        )

                        # Convert to tensor for compatibility with existing code
                        mask_tensor = torch.from_numpy(refined_mask.astype(np.float32)).float().unsqueeze(0).unsqueeze(0).to(self.device)
                        refined_masks.append(mask_tensor)

                        # Force garbage collection every few objects when using CUDA
                        if self.use_cuda and (i + 1) % 3 == 0:
                            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error in segmentation: {e}")

        return boxes_px, refined_masks


class MaskUtility:
    @staticmethod
    def remove_small_regions(
        mask: np.ndarray,
        min_size_ratio: float = 0.01,
        min_size_abs: int = 10,
    ) -> np.ndarray:
        """Optimized version to remove small disconnected regions."""
        # Early exit for empty masks
        if not np.any(mask):
            return mask

        labeled_mask, num_features = ndimage.label(mask)

        if num_features <= 1:  # Only one region, return as is
            return mask

        # Compute component sizes efficiently
        indices, counts = np.unique(labeled_mask[labeled_mask > 0], return_counts=True)

        if len(counts) == 0:
            return mask

        largest_component_size = counts.max()
        min_size = max(min_size_abs, int(largest_component_size * min_size_ratio))

        # Create a LUT for fast relabeling
        keep_label = np.zeros(num_features + 1, dtype=bool)
        keep_label[indices] = counts >= min_size

        # Apply the LUT (much faster than creating a mask and indexing)
        return keep_label[labeled_mask]

    @staticmethod
    def refine_mask(
        mask: np.ndarray,
        fill_holes: bool = True,
        smooth: bool = True,
        smooth_pixels: int = 0,  # Add this parameter
        dilate_pixels: int | None = None,
        area: int | None = None,
        remove_small: bool = True,
        min_size_ratio: float = 0.01,
        min_size_abs: int = 10,
    ) -> np.ndarray:
        """Refine a segmentation mask with subtle improvements."""
        mask_bin = mask.astype(bool)

        if fill_holes:
            mask_bin = ndimage.binary_fill_holes(mask_bin)
            kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)
            neighbors = ndimage.convolve(
                mask_bin.astype(int), kernel, mode="constant"
            )
            mask_bin = np.logical_or(mask_bin, neighbors >= 3)

        if remove_small:
            mask_bin = MaskUtility.remove_small_regions(
                mask_bin, min_size_ratio, min_size_abs
            )

        if smooth:
            # Use smooth_pixels if provided and non-zero, otherwise default behavior
            if smooth_pixels > 0:
                # Convert smooth_pixels to sigma value (divide by a scaling factor)
                # Values 1-10 map to sigma range of approximately 0.2-2.0
                sigma = smooth_pixels / 5.0
                mask_float = ndimage.gaussian_filter(mask_bin.astype(float), sigma=sigma)
            else:
                # Use default built-in smoothing
                mask_float = ndimage.gaussian_filter(mask_bin.astype(float), sigma=0.7)

            mask_bin = mask_float > 0.3

        if dilate_pixels is None and area is not None:
            dilate_pixels = int((area * 0.00001) + 0.5)
        elif dilate_pixels is None:
            dilate_pixels = 4

        if dilate_pixels > 0:
            y, x = np.ogrid[-dilate_pixels:dilate_pixels+1, -dilate_pixels:dilate_pixels+1]
            disk = x * x + y * y <= dilate_pixels * dilate_pixels
            mask_bin = ndimage.binary_dilation(mask_bin, structure=disk)

        if remove_small:
            mask_bin = MaskUtility.remove_small_regions(
                mask_bin, min_size_ratio, min_size_abs
            )

        return mask_bin

    @staticmethod
    def create_binary_mask(
        masks: list[np.ndarray],
        image_shape: tuple[int, ...],
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Create a combined binary mask from multiple segmentation masks."""
        if not masks:
            # Create empty mask directly as uint8 (avoid bool→uint8 conversion)
            binary_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        elif len(masks) == 1:
            # Fast path for single mask (common case)
            binary_mask = masks[0].astype(np.uint8) * 255
        else:
            # Use in-place operations to reduce memory allocation
            combined = masks[0].copy()
            for mask in masks[1:]:
                np.logical_or(combined, mask, out=combined)
            binary_mask = combined.astype(np.uint8) * 255

        if save_path:
            cv2.imwrite(str(save_path), binary_mask)

        return binary_mask

    @staticmethod
    def combine_tensor_masks(masks: list[torch.Tensor], shape: tuple[int, int] | None = None) -> torch.Tensor | None:
        """
        Combine multiple tensor masks into a single mask.

        Args:
            masks (List[torch.Tensor]): List of mask tensors
            shape (Tuple[int, int], optional): Shape of the output mask (height, width)

        Returns:
            torch.Tensor: Combined mask tensor
        """
        if not masks:
            return None

        if len(masks) == 1:
            return masks[0]

        # Start with the first mask
        combined = masks[0].clone()

        # Combine with other masks using maximum operation
        for mask in masks[1:]:
            combined = torch.max(combined, mask)

        return combined


class ImageUtility:
    @staticmethod
    def numpy_to_pil(np_array: np.ndarray) -> Image.Image:
        """Convert a numpy array to a PIL Image."""
        if np_array.dtype != np.uint8:
            np_array = (np_array * 255).astype(np.uint8)
        return Image.fromarray(np_array)

    @staticmethod
    def find_images(directory: str | Path) -> list[Path]:
        """Find all image files in the given directory, excluding mask files."""
        directory = Path(directory) if isinstance(directory, str) else directory

        # Common image extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

        # Use a set of canonical paths to track unique files
        seen_canonical_paths = set()
        unique_images = []

        for ext in image_extensions:
            pattern = f"*{ext}"
            for img_path in directory.glob(pattern):
                # Skip mask files (files with "-masklabel" in their name)
                if "-masklabel" in img_path.stem:
                    continue

                # Get canonical path to handle case insensitivity on Windows
                canonical_path = str(img_path).lower()
                if canonical_path not in seen_canonical_paths:
                    seen_canonical_paths.add(canonical_path)
                    unique_images.append(img_path)

        sorted_images = sorted(unique_images)
        logger.info(f"Found {len(sorted_images)} unique images")
        return sorted_images


class MoondreamSAMMaskModel(BaseImageMaskModel):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        sam2_model_size: str = "large",  # "tiny", "small", "base-plus", or "large"
        moondream_model_revision: str = "05d640e6da70c37b2473e0db8fef0233c0709ce4",
    ):
        """
        Initialize the MoondreamSAM mask model using SAM2 from Hugging Face.

        Args:
            device (torch.device): Device to use (CPU/CUDA)
            dtype (torch.dtype): Data type for models
            sam2_model_size (str): SAM2 model size - "tiny", "small", "base-plus", "large"
            moondream_model_revision (str): Moondream2 model revision
        """
        self.device = device
        self.dtype = dtype
        self.use_cuda = device.type == "cuda"
        self.model_manager = None  # Will be set when loaded through model_manager

        # Map model size to proper HF model name
        self.sam2_model_map = {
            "tiny": "facebook/sam2-hiera-tiny",
            "small": "facebook/sam2-hiera-small",
            "base-plus": "facebook/sam2-hiera-base-plus",
            "large": "facebook/sam2-hiera-large"
        }

        self.sam2_model_name = self.sam2_model_map.get(sam2_model_size, "facebook/sam2-hiera-large")
        self.moondream_model_revision = moondream_model_revision

        # Initialize components using enhanced implementation
        self.detector = ObjectDetector(
            device=device,
            dtype=dtype,
            model_revision=moondream_model_revision
        )

        self.segmenter = Segmenter(
            device=device,
            sam2_model_name=self.sam2_model_name
        )

    def _load_models(self):
        """Load both Moondream2 and SAM2 models."""
        self.detector.load(model_manager=self.model_manager)
        self.segmenter.load()

    def mask_image(
        self,
        filename: str,
        prompts: list[str],
        mode: str = 'fill',
        alpha: float = 1.0,
        threshold: float = 0.3,
        smooth_pixels: int = 5,
        expand_pixels: int = 10
    ):
        """
        Masks a sample based on detected objects and SAM2 segmentation.

        Args:
            filename (str): Path to the image file
            prompts (list[str]): List of object prompts to detect and mask
            mode (str): Masking mode ('fill' or other)
            alpha (float): Alpha blending value
            threshold (float): Threshold for mask activation
            smooth_pixels (int): Pixels for smoothing the mask
            expand_pixels (int): Pixels to expand the mask
        """
        # Create a mask sample
        mask_sample = MaskSample(filename, self.device)

        logger.info(f"Processing {filename}")
        logger.info(f"Mask will be saved to: {mask_sample.mask_filename}")

        # Skip if mode is fill and mask already exists
        if mode == 'fill' and os.path.exists(mask_sample.mask_filename):
            logger.info(f"Skipping {filename} as mask already exists")
            return

        # Ensure models are loaded
        self._load_models()

        # Load image
        image = mask_sample.get_image()
        logger.info(f"Image loaded, size: {image.size}")

        # Process each prompt for detection
        combined_mask = None

        for prompt in prompts:
            logger.info(f"Processing prompt: '{prompt}' for {filename}")

            # Detect objects using Moondream2
            try:
                np_image, detected_objects, img_dimensions = self.detector.detect(
                    image_path=filename,
                    prompt=prompt,
                    max_objects=5  # Limit to 5 objects per prompt
                )

                if np_image is None or not detected_objects:
                    logger.info(f"No objects detected for prompt: '{prompt}'")
                    continue

                logger.info(f"Detected {len(detected_objects)} objects for prompt: '{prompt}'")

                _, object_masks = self.segmenter.segment(
                    np_image=np_image,
                    detected_objects=detected_objects,
                    image_dimensions=img_dimensions,
                    threshold=threshold,
                    smooth_pixels=smooth_pixels,
                    expand_pixels=expand_pixels
                )

                if not object_masks:
                    logger.info(f"No valid masks generated for prompt: '{prompt}'")
                    continue

                # Combine masks for this prompt
                prompt_mask = MaskUtility.combine_tensor_masks(object_masks)

                # Add to combined mask across all prompts
                combined_mask = prompt_mask if combined_mask is None else torch.max(combined_mask, prompt_mask)

            except Exception as e:
                logger.error(f"Error processing prompt '{prompt}': {str(e)}")
                import traceback
                traceback.print_exc()

        # If no masks were created, return
        if combined_mask is None:
            logger.info(f"No masks generated for {filename} with prompts {prompts}")
            return
        else:
            # Print some stats about the combined mask
            mask_stats = {
                "min": combined_mask.min().item(),
                "max": combined_mask.max().item(),
                "mean": combined_mask.mean().item(),
                "shape": list(combined_mask.shape)
            }
            logger.debug(f"Final mask stats: {mask_stats}")

        # Apply the mask to the sample
        mask_sample.apply_mask(mode, combined_mask, alpha, False)

        # Save the mask
        mask_sample.save_mask()
        logger.info(f"Saved mask for {filename}")
