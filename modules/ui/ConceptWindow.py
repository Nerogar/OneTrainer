import os
import random

from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.BalancingStrategy import BalancingStrategy
from modules.util.ui import components
from modules.util.ui.UIState import UIState

from mgds.LoadingPipeline import LoadingPipeline
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch
from torchvision.transforms import functional

import customtkinter as ctk
from PIL import Image


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


class ConceptWindow(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            concept: ConceptConfig,
            ui_state: UIState,
            image_ui_state: UIState,
            text_ui_state: UIState,
            *args, **kwargs,
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.concept = concept
        self.ui_state = ui_state
        self.image_ui_state = image_ui_state
        self.text_ui_state = text_ui_state

        self.image_preview_file_index = 0

        self.title("概念")
        self.geometry("800x630")
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        tabview = ctk.CTkTabview(self)
        tabview.grid(row=0, column=0, sticky="nsew")

        self.general_tab = self.__general_tab(tabview.add("基础设置"), concept)
        self.image_augmentation_tab = self.__image_augmentation_tab(tabview.add("图像增强"))
        self.text_augmentation_tab = self.__text_augmentation_tab(tabview.add("文本增强"))

        components.button(self, 1, 0, "ok", self.__ok)

    def __general_tab(self, master, concept: ConceptConfig):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)

        # name
        components.label(frame, 0, 0, "名称",
                         tooltip="概念的名称")
        components.entry(frame, 0, 1, self.ui_state, "name")

        # enabled
        components.label(frame, 1, 0, "启用",
                         tooltip="启用或禁用此概念")
        components.switch(frame, 1, 1, self.ui_state, "enabled")

        # validation_concept
        components.label(frame, 2, 0, "验证概念",
                         tooltip="使用概念作为验证而不是训练")
        components.switch(frame, 2, 1, self.ui_state, "validation_concept")

        # path
        components.label(frame, 3, 0, "路径",
                         tooltip="训练数据所在的路径")
        components.dir_entry(frame, 3, 1, self.ui_state, "path")

        # prompt source
        components.label(frame, 4, 0, "提示来源",
                         tooltip="训练期间使用的提示的来源。选择“从单个文本文件”时，请选择包含提示列表的文本文件")
        prompt_path_entry = components.file_entry(frame, 4, 2, self.text_ui_state, "prompt_path")

        def set_prompt_path_entry_enabled(option: str):
            if option == 'concept':
                for child in prompt_path_entry.children.values():
                    child.configure(state="normal")
            else:
                for child in prompt_path_entry.children.values():
                    child.configure(state="disabled")

        components.options_kv(frame, 4, 1, [
            ("从每个样本的文本文件获取", 'sample'),
            ("从单个文本文件获取", 'concept'),
            ("从图像文件名获取", 'filename'),
        ], self.text_ui_state, "prompt_source", command=set_prompt_path_entry_enabled)
        set_prompt_path_entry_enabled(concept.text.prompt_source)

        # include subdirectories
        components.label(frame, 5, 0, "包含子目录",
                         tooltip="将子目录中的图像包含到数据集中")
        components.switch(frame, 5, 1, self.ui_state, "include_subdirectories")

        # image variations
        components.label(frame, 6, 0, "图像变体",
                         tooltip="如果启用了潜在缓存，要缓存的不同图像版本的数量。")
        components.entry(frame, 6, 1, self.ui_state, "image_variations")

        # text variations
        components.label(frame, 7, 0, "文本变体",
                         tooltip="如果启用了潜在缓存，要缓存的不同文本版本的数量。")
        components.entry(frame, 7, 1, self.ui_state, "text_variations")

        # balancing
        components.label(frame, 8, 0, "平衡",
                         tooltip="训练期间使用的样本数量。使用重复来乘以概念，或者使用样本在每个时期中指定使用的确切样本数量。")
        components.entry(frame, 8, 1, self.ui_state, "balancing")
        components.options(frame, 8, 2, [str(x) for x in list(BalancingStrategy)], self.ui_state, "balancing_strategy")

        # loss weight
        components.label(frame, 9, 0, "损失权重",
                         tooltip="此概念的损失乘数。")
        components.entry(frame, 9, 1, self.ui_state, "loss_weight")

        frame.pack(fill="both", expand=1)
        return frame

    def __image_augmentation_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=0)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        # header
        components.label(frame, 0, 1, "随机",
                         tooltip="使用随机值启用此增强")
        components.label(frame, 0, 2, "固定",
                         tooltip="使用固定值启用此增强")

        # crop jitter
        components.label(frame, 1, 0, "裁剪抖动",
                         tooltip="启用样本的随机裁剪")
        components.switch(frame, 1, 1, self.image_ui_state, "enable_crop_jitter")

        # random flip
        components.label(frame, 2, 0, "随机翻转",
                         tooltip="在训练期间随机翻转样本")
        components.switch(frame, 2, 1, self.image_ui_state, "enable_random_flip")
        components.switch(frame, 2, 2, self.image_ui_state, "enable_fixed_flip")

        # random rotation
        components.label(frame, 3, 0, "随机旋转",
                         tooltip="在训练期间随机旋转样本")
        components.switch(frame, 3, 1, self.image_ui_state, "enable_random_rotate")
        components.switch(frame, 3, 2, self.image_ui_state, "enable_fixed_rotate")
        components.entry(frame, 3, 3, self.image_ui_state, "random_rotate_max_angle")

        # random brightness
        components.label(frame, 4, 0, "随机亮度",
                         tooltip="在训练期间随机调整样本的亮度")
        components.switch(frame, 4, 1, self.image_ui_state, "enable_random_brightness")
        components.switch(frame, 4, 2, self.image_ui_state, "enable_fixed_brightness")
        components.entry(frame, 4, 3, self.image_ui_state, "random_brightness_max_strength")

        # random contrast
        components.label(frame, 5, 0, "随机对比度",
                         tooltip="在训练期间随机调整样本的对比度")
        components.switch(frame, 5, 1, self.image_ui_state, "enable_random_contrast")
        components.switch(frame, 5, 2, self.image_ui_state, "enable_fixed_contrast")
        components.entry(frame, 5, 3, self.image_ui_state, "random_contrast_max_strength")

        # random saturation
        components.label(frame, 6, 0, "随机饱和度",
                         tooltip="在训练期间随机调整样本的饱和度")
        components.switch(frame, 6, 1, self.image_ui_state, "enable_random_saturation")
        components.switch(frame, 6, 2, self.image_ui_state, "enable_fixed_saturation")
        components.entry(frame, 6, 3, self.image_ui_state, "random_saturation_max_strength")

        # random hue
        components.label(frame, 7, 0, "随机色调",
                         tooltip="在训练期间随机调整样本的色调")
        components.switch(frame, 7, 1, self.image_ui_state, "enable_random_hue")
        components.switch(frame, 7, 2, self.image_ui_state, "enable_fixed_hue")
        components.entry(frame, 7, 3, self.image_ui_state, "random_hue_max_strength")

        # random circular mask shrink
        components.label(frame, 8, 0, "圆形掩码生成",
                         tooltip="自动为掩码训练创建圆形掩码")
        components.switch(frame, 8, 1, self.image_ui_state, "enable_random_circular_mask_shrink")

        # random rotate and crop
        components.label(frame, 9, 0, "随机旋转和裁剪",
                         tooltip="随机旋转训练样本并裁剪到掩码区域")
        components.switch(frame, 9, 1, self.image_ui_state, "enable_random_mask_rotate_crop")

        # circular mask generation
        components.label(frame, 10, 0, "分辨率覆盖",
                         tooltip="覆盖此概念的分辨率。可以选择指定多个分辨率（用逗号分隔），或以 <宽度>x<高度> 的格式指定单个精确分辨率")
        components.switch(frame, 10, 2, self.image_ui_state, "enable_resolution_override")
        components.entry(frame, 10, 3, self.image_ui_state, "resolution_override")

        # image
        preview = self.__get_preview_image()
        self.image = ctk.CTkImage(
            light_image=preview,
            size=preview.size,
        )
        image_label = ctk.CTkLabel(master=frame, text="", image=self.image, height=300, width=300)
        image_label.grid(row=0, column=4, rowspan=6)

        # refresh preview
        update_button_frame = ctk.CTkFrame(master=frame, corner_radius=0, fg_color="transparent")
        update_button_frame.grid(row=6, column=4, sticky="nsew")
        update_button_frame.grid_columnconfigure(1, weight=1)

        prev_preview_button = components.button(update_button_frame, 0, 0, "<", command=self.__prev_image_preview)
        components.button(update_button_frame, 0, 1, "更新预览", command=self.__update_image_preview)
        next_preview_button = components.button(update_button_frame, 0, 2, ">", command=self.__next_image_preview)

        prev_preview_button.configure(width=40)
        next_preview_button.configure(width=40)

        frame.pack(fill="both", expand=1)
        return frame

    def __text_augmentation_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=0)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        # tag shuffling
        components.label(frame, 0, 0, "随机打乱",
                         tooltip="启用随机打乱")
        components.switch(frame, 0, 1, self.text_ui_state, "enable_tag_shuffling")

        # keep tag count
        components.label(frame, 1, 0, "标签分隔符",
                         tooltip="标签之间的分隔符")
        components.entry(frame, 1, 1, self.text_ui_state, "tag_delimiter")

        # keep tag count
        components.label(frame, 2, 0, "固定标签",
                         tooltip="在随机打乱 tokens 时，保留前 N 个不变。")
        components.entry(frame, 2, 1, self.text_ui_state, "keep_tags_count")

        frame.pack(fill="both", expand=1)
        return frame

    def __prev_image_preview(self):
        self.image_preview_file_index = max(self.image_preview_file_index - 1, 0)
        self.__update_image_preview()

    def __next_image_preview(self):
        self.image_preview_file_index += 1
        self.__update_image_preview()

    def __update_image_preview(self):
        preview = self.__get_preview_image()
        self.image.configure(light_image=preview, size=preview.size)

    def __get_preview_image(self):
        preview_image_path = "resources/icons/icon.png"

        file_index = -1
        if os.path.isdir(self.concept.path):
            for path in os.scandir(self.concept.path):
                extension = os.path.splitext(path)[1]
                if path.is_file() \
                        and path_util.is_supported_image_extension(extension) \
                        and not path.name.endswith("-masklabel.png"):
                    preview_image_path = path_util.canonical_join(self.concept.path, path.name)

                    file_index += 1
                    if file_index == self.image_preview_file_index:
                        break

        image = Image.open(preview_image_path).convert("RGB")
        image_tensor = functional.to_tensor(image)

        splitext = os.path.splitext(os.path.basename(preview_image_path))
        preview_mask_path = path_util.canonical_join(self.concept.path, splitext[0] + "-masklabel.png")
        if not os.path.isfile(preview_mask_path):
            preview_mask_path = None

        if preview_mask_path:
            mask = Image.open(preview_mask_path).convert("L")
            mask_tensor = functional.to_tensor(mask)
        else:
            mask_tensor = torch.ones((1, image_tensor.shape[1], image_tensor.shape[2]))

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
        })

        circular_mask_shrink = RandomCircularMaskShrink(mask_name='mask', shrink_probability=1.0, shrink_factor_min=0.2, shrink_factor_max=1.0, enabled_in_name='enable_random_circular_mask_shrink')
        random_mask_rotate_crop = RandomMaskRotateCrop(mask_name='mask', additional_names=['image'], min_size=512, min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20, enabled_in_name='enable_random_mask_rotate_crop')
        random_flip = RandomFlip(names=['image', 'mask'], enabled_in_name='enable_random_flip', fixed_enabled_in_name='enable_fixed_flip')
        random_rotate = RandomRotate(names=['image', 'mask'], enabled_in_name='enable_random_rotate', fixed_enabled_in_name='enable_fixed_rotate', max_angle_in_name='random_rotate_max_angle')
        random_brightness = RandomBrightness(names=['image'], enabled_in_name='enable_random_brightness', fixed_enabled_in_name='enable_fixed_brightness', max_strength_in_name='random_brightness_max_strength')
        random_contrast = RandomContrast(names=['image'], enabled_in_name='enable_random_contrast', fixed_enabled_in_name='enable_fixed_contrast', max_strength_in_name='random_contrast_max_strength')
        random_saturation = RandomSaturation(names=['image'], enabled_in_name='enable_random_saturation', fixed_enabled_in_name='enable_fixed_saturation', max_strength_in_name='random_saturation_max_strength')
        random_hue = RandomHue(names=['image'], enabled_in_name='enable_random_hue', fixed_enabled_in_name='enable_fixed_hue', max_strength_in_name='random_hue_max_strength')
        output_module = OutputPipelineModule(['image', 'mask'])

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

        mask_tensor = torch.clamp(mask_tensor, 0.3, 1)
        image_tensor = image_tensor * mask_tensor

        image = functional.to_pil_image(image_tensor)

        image.thumbnail((300, 300))

        return image

    def __ok(self):
        self.destroy()
