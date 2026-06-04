import os
import pathlib
import platform
import random
import threading
import time
import traceback

from modules.util import concept_stats, path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.image_util import load_image

from mgds.LoadingPipeline import LoadingPipeline
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.CapitalizeTags import CapitalizeTags
from mgds.pipelineModules.DropTags import DropTags
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomCircularMaskShrink import (
    RandomCircularMaskShrink,
)
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
from mgds.pipelineModules.ShuffleTags import ShuffleTags
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch
from torchvision.transforms import functional

import huggingface_hub
from PIL import Image


class ConceptWindowController:
    def __init__(self, train_config: TrainConfig, concept: ConceptConfig):
        self.train_config = train_config
        self.concept = concept
        self.cancel_scan_flag = threading.Event()
        self.scan_thread = None

    @staticmethod
    def get_concept_path(path: str) -> str | None:
        if os.path.isdir(path):
            return path
        try:
            #don't download, only check if available locally:
            return huggingface_hub.snapshot_download(repo_id=path, repo_type="dataset", local_files_only=True)
        except Exception:
            return None

    def download_dataset(self):
        try:
            if self.train_config.secrets.huggingface_token != "":
                huggingface_hub.login(token=self.train_config.secrets.huggingface_token)
            huggingface_hub.snapshot_download(repo_id=self.concept.path, repo_type="dataset")
        except Exception:
            traceback.print_exc()

    def download_dataset_threaded(self):
        download_thread = threading.Thread(target=self.download_dataset, daemon=True)
        download_thread.start()

    def _read_text_file_for_preview(self, file_path: str, preview_augmentations: bool) -> str:
        empty_msg = "[Empty prompt]"
        try:
            with open(file_path, "r") as f:
                if preview_augmentations:
                    lines = [line.strip() for line in f if line.strip()]
                    return random.choice(lines) if lines else empty_msg
                content = f.read().strip()
                return content if content else empty_msg
        except FileNotFoundError:
            return "File not found, please check the path"
        except IsADirectoryError:
            return "[Provided path is a directory, please correct the caption path]"
        except PermissionError:
            if platform.system() == "Windows":
                return "[Permission denied, please check the file permissions or Windows Defender settings]"
            else:
                return "[Permission denied, please check the file permissions]"
        except UnicodeDecodeError:
            return "[Invalid file encoding. This should not happen, please report this issue]"

    def get_preview_image(self, image_preview_file_index: int, preview_augmentations: bool):
        preview_image_path = "resources/icons/icon.png"
        file_index = -1
        glob_pattern = "**/*.*" if self.concept.include_subdirectories else "*.*"

        concept_path = self.get_concept_path(self.concept.path)
        if concept_path:
            for path in pathlib.Path(concept_path).glob(glob_pattern):
                if any(part.startswith('.') for part in path.relative_to(concept_path).parent.parts):
                    continue
                extension = os.path.splitext(path)[1]
                if path.is_file() and path_util.is_supported_image_extension(extension) \
                        and not path.name.endswith("-masklabel.png") and not path.name.endswith("-condlabel.png"):
                    preview_image_path = path_util.canonical_join(concept_path, path)
                    file_index += 1
                    if file_index == image_preview_file_index:
                        break

        image = load_image(preview_image_path, 'RGB')
        image_tensor = functional.to_tensor(image)

        splitext = os.path.splitext(preview_image_path)
        preview_mask_path = path_util.canonical_join(splitext[0] + "-masklabel.png")
        if not os.path.isfile(preview_mask_path):
            preview_mask_path = None

        if preview_mask_path:
            mask = Image.open(preview_mask_path).convert("L")
            mask_tensor = functional.to_tensor(mask)
        else:
            mask_tensor = torch.ones((1, image_tensor.shape[1], image_tensor.shape[2]))

        source = self.concept.text.prompt_source
        preview_p = pathlib.Path(preview_image_path)
        if source == "filename":
            prompt_output = preview_p.stem or "[Empty prompt]"
        else:
            file_map = {
                "sample": preview_p.with_suffix(".txt"),
                "concept": pathlib.Path(self.concept.text.prompt_path) if self.concept.text.prompt_path else None,
            }
            file_path = file_map.get(source)
            prompt_output = self._read_text_file_for_preview(str(file_path), preview_augmentations) if file_path else "[Empty prompt]"

        modules = []
        if preview_augmentations:
            input_module = InputPipelineModule({
                'true': True,
                'image': image_tensor,
                'mask': mask_tensor,
                'enable_random_flip': self.concept.image.enable_random_flip,
                'enable_fixed_flip': self.concept.image.enable_fixed_flip,
                'enable_random_rotate': self.concept.image.enable_random_rotate,
                'enable_fixed_rotate': self.concept.image.enable_fixed_rotate,
                'random_rotate_max_angle': self.concept.image.random_rotate_max_angle,
                'enable_random_brightness': self.concept.image.enable_random_brightness,
                'enable_fixed_brightness': self.concept.image.enable_fixed_brightness,
                'random_brightness_max_strength': self.concept.image.random_brightness_max_strength,
                'enable_random_contrast': self.concept.image.enable_random_contrast,
                'enable_fixed_contrast': self.concept.image.enable_fixed_contrast,
                'random_contrast_max_strength': self.concept.image.random_contrast_max_strength,
                'enable_random_saturation': self.concept.image.enable_random_saturation,
                'enable_fixed_saturation': self.concept.image.enable_fixed_saturation,
                'random_saturation_max_strength': self.concept.image.random_saturation_max_strength,
                'enable_random_hue': self.concept.image.enable_random_hue,
                'enable_fixed_hue': self.concept.image.enable_fixed_hue,
                'random_hue_max_strength': self.concept.image.random_hue_max_strength,
                'enable_random_circular_mask_shrink': self.concept.image.enable_random_circular_mask_shrink,
                'enable_random_mask_rotate_crop': self.concept.image.enable_random_mask_rotate_crop,

                'prompt' : prompt_output,
                'tag_dropout_enable' : self.concept.text.tag_dropout_enable,
                'tag_dropout_probability' : self.concept.text.tag_dropout_probability,
                'tag_dropout_mode' : self.concept.text.tag_dropout_mode,
                'tag_dropout_special_tags' : self.concept.text.tag_dropout_special_tags,
                'tag_dropout_special_tags_mode' : self.concept.text.tag_dropout_special_tags_mode,
                'tag_delimiter' : self.concept.text.tag_delimiter,
                'keep_tags_count' : self.concept.text.keep_tags_count,
                'tag_dropout_special_tags_regex' : self.concept.text.tag_dropout_special_tags_regex,
                'caps_randomize_enable' : self.concept.text.caps_randomize_enable,
                'caps_randomize_probability' : self.concept.text.caps_randomize_probability,
                'caps_randomize_mode' : self.concept.text.caps_randomize_mode,
                'caps_randomize_lowercase' : self.concept.text.caps_randomize_lowercase,
                'enable_tag_shuffling' : self.concept.text.enable_tag_shuffling,
            })

            circular_mask_shrink = RandomCircularMaskShrink(mask_name='mask', shrink_probability=1.0, shrink_factor_min=0.2, shrink_factor_max=1.0, enabled_in_name='enable_random_circular_mask_shrink')
            random_mask_rotate_crop = RandomMaskRotateCrop(mask_name='mask', additional_names=['image'], min_size=512, min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20, enabled_in_name='enable_random_mask_rotate_crop')
            random_flip = RandomFlip(names=['image', 'mask'], enabled_in_name='enable_random_flip', fixed_enabled_in_name='enable_fixed_flip')
            random_rotate = RandomRotate(names=['image', 'mask'], enabled_in_name='enable_random_rotate', fixed_enabled_in_name='enable_fixed_rotate', max_angle_in_name='random_rotate_max_angle')
            random_brightness = RandomBrightness(names=['image'], enabled_in_name='enable_random_brightness', fixed_enabled_in_name='enable_fixed_brightness', max_strength_in_name='random_brightness_max_strength')
            random_contrast = RandomContrast(names=['image'], enabled_in_name='enable_random_contrast', fixed_enabled_in_name='enable_fixed_contrast', max_strength_in_name='random_contrast_max_strength')
            random_saturation = RandomSaturation(names=['image'], enabled_in_name='enable_random_saturation', fixed_enabled_in_name='enable_fixed_saturation', max_strength_in_name='random_saturation_max_strength')
            random_hue = RandomHue(names=['image'], enabled_in_name='enable_random_hue', fixed_enabled_in_name='enable_fixed_hue', max_strength_in_name='random_hue_max_strength')
            drop_tags = DropTags(text_in_name='prompt', enabled_in_name='tag_dropout_enable', probability_in_name='tag_dropout_probability', dropout_mode_in_name='tag_dropout_mode',
                                special_tags_in_name='tag_dropout_special_tags', special_tag_mode_in_name='tag_dropout_special_tags_mode', delimiter_in_name='tag_delimiter',
                                keep_tags_count_in_name='keep_tags_count', text_out_name='prompt', regex_enabled_in_name='tag_dropout_special_tags_regex')
            caps_randomize = CapitalizeTags(text_in_name='prompt', enabled_in_name='caps_randomize_enable', probability_in_name='caps_randomize_probability',
                                            capitalize_mode_in_name='caps_randomize_mode', delimiter_in_name='tag_delimiter', convert_lowercase_in_name='caps_randomize_lowercase', text_out_name='prompt')
            shuffle_tags = ShuffleTags(text_in_name='prompt', enabled_in_name='enable_tag_shuffling', delimiter_in_name='tag_delimiter', keep_tags_count_in_name='keep_tags_count', text_out_name='prompt')
            output_module = OutputPipelineModule(['image', 'mask', 'prompt'])

            modules = [
                input_module,
                circular_mask_shrink,
                random_mask_rotate_crop,
                random_flip,
                random_rotate,
                random_brightness,
                random_contrast,
                random_saturation,
                random_hue,
                drop_tags,
                caps_randomize,
                shuffle_tags,
                output_module,
            ]

            pipeline = LoadingPipeline(
                device=torch.device('cpu'),
                modules=modules,
                batch_size=1,
                seed=random.randint(0, 2**30),
                state=None,
                initial_epoch=0,
                initial_index=0,
            )

            data = pipeline.__next__()
            image_tensor = data['image']
            mask_tensor = data['mask']
            prompt_output = data['prompt']

        filename_output = os.path.basename(preview_image_path)

        mask_tensor = torch.clamp(mask_tensor, 0.3, 1)
        image_tensor = image_tensor * mask_tensor

        image = functional.to_pil_image(image_tensor)

        image.thumbnail((300, 300))

        return image, filename_output, prompt_output

    def get_concept_stats(self, view, advanced_checks: bool, wait_time: float):
        start_time = time.perf_counter()
        last_update = time.perf_counter()
        self.cancel_scan_flag.clear()
        view.components.call_after(view.concept_stats_tab, 0, view._disable_scan_buttons)
        concept_path = self.get_concept_path(self.concept.path)

        if not concept_path:
           print(f"Unable to get statistics for concept path: {self.concept.path}")
           view.components.call_after(view.concept_stats_tab, 0, view._enable_scan_buttons)
           return
        subfolders = [concept_path]

        stats_dict = concept_stats.init_concept_stats(advanced_checks)
        for path in subfolders:
            if self.cancel_scan_flag.is_set() or time.perf_counter() - start_time > wait_time:
                break
            stats_dict = concept_stats.folder_scan(path, stats_dict, advanced_checks, self.concept, start_time, wait_time, self.cancel_scan_flag)
            if self.concept.include_subdirectories and not self.cancel_scan_flag.is_set():     #add all subfolders of current directory to for loop
                subfolders.extend([f for f in os.scandir(path) if f.is_dir() and not f.name.startswith('.')])
            self.concept.concept_stats = stats_dict
            #update GUI approx every half second
            if time.perf_counter() > (last_update + 0.5):
                last_update = time.perf_counter()
                view.components.call_after(view.concept_stats_tab, 0, lambda: view._update_concept_stats(self))

        self.cancel_scan_flag.clear()
        view.components.call_after(view.concept_stats_tab, 0, view._enable_scan_buttons)
        view.components.call_after(view.concept_stats_tab, 0, lambda: view._update_concept_stats(self))

    def get_concept_stats_threaded(self, view, advanced_checks: bool, waittime: float):
        self.scan_thread = threading.Thread(target=self.get_concept_stats, args=[view, advanced_checks, waittime], daemon=True)
        self.scan_thread.start()

    def auto_update_concept_stats(self, view):
        try:
            view._update_concept_stats(self)      #load stats from config if available, else raises KeyError
            if self.concept.concept_stats["file_size"] == 0:  #force rescan if empty
                raise KeyError
        except KeyError:
            concept_path = self.get_concept_path(self.concept.path)
            if concept_path:
                self.get_concept_stats(view, False, 2)    #force rescan if config is empty, timeout of 2 sec
                if self.concept.concept_stats["processing_time"] < 0.1:
                    self.get_concept_stats(view, True, 2)    #do advanced scan automatically if basic took <0.1s


class InputPipelineModule(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def length(self) -> int:
        return 1

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return list(self.data.keys())

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return self.data
