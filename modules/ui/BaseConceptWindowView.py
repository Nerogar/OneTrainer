import fractions
import math

from modules.util import path_util
from modules.util.enum.BalancingStrategy import BalancingStrategy
from modules.util.enum.ConceptType import ConceptType


class BaseConceptWindowView:
    def __init__(self, components):
        self.components = components
        self.bucket_ax = None
        self.text_color = None
        self.canvas = None

    def build_general_tab(self, frame, controller, ui_state, text_ui_state):
        # name
        self.components.label(frame, 0, 0, "名称",
                         tooltip="数据集名称")
        self.components.entry(frame, 0, 1, ui_state, "name")

        # enabled
        self.components.label(frame, 1, 0, "启用",
                         tooltip="启用或禁用此数据集")
        self.components.switch(frame, 1, 1, ui_state, "enabled")

        # concept type
        self.components.label(frame, 2, 0, "数据集类型",
                         tooltip="STANDARD: Standard finetuning with the sample as training target\n"
                                 "VALIDATION: Use concept for validation instead of training\n"
                                 "PRIOR_PREDICTION: Use the sample to make a prediction using the model as it was before training. This prediction is then used as the training target "
                                 "for the model in training. This can be used as regularisation and to preserve prior model knowledge while finetuning the model on other concepts. "
                                 "仅对LoRA实现",
                         wide_tooltip=True)
        self.components.options(frame, 2, 1, [str(x) for x in list(ConceptType)], ui_state, "type")

        # path
        self.components.label(frame, 3, 0, "路径",
                         tooltip="训练数据所在路径")
        self.components.path_entry(frame, 3, 1, ui_state, "path", mode="dir")
        self.components.button(frame, 3, 2, text="download now", command=controller.download_dataset_threaded,
                          tooltip="从Huggingface下载数据集用于预览和统计")

        # prompt source
        self.components.label(frame, 4, 0, "提示词来源",
                         tooltip="The source for prompts used during training. When selecting \"From single text file\", select a text file that contains a list of prompts")
        prompt_path_entry = self.components.path_entry(frame, 4, 2, text_ui_state, "prompt_path", mode="file")

        def set_prompt_path_entry_enabled(option: str):
            self.components.set_widget_enabled(prompt_path_entry, option == 'concept')

        self.components.options_kv(frame, 4, 1, [
            ("从每样本文本文件", 'sample'),
            ("从单个文本文件", 'concept'),
            ("从图像文件名", 'filename'),
        ], text_ui_state, "prompt_source", command=set_prompt_path_entry_enabled)
        set_prompt_path_entry_enabled(controller.concept.text.prompt_source)

        # include subdirectories
        self.components.label(frame, 5, 0, "包含子目录",
                         tooltip="将子目录中的图像包含到数据集中")
        self.components.switch(frame, 5, 1, ui_state, "include_subdirectories")

        # image variations
        self.components.label(frame, 6, 0, "图像变体",
                         tooltip="潜在缓存的图像版本数")
        self.components.entry(frame, 6, 1, ui_state, "image_variations")

        # text variations
        self.components.label(frame, 7, 0, "文本变体",
                         tooltip="潜在缓存的文本版本数")
        self.components.entry(frame, 7, 1, ui_state, "text_variations")

        # balancing
        self.components.label(frame, 8, 0, "平衡策略",
                         tooltip="训练使用的样本数，用repeats倍乘或samples指定精确数")
        self.components.entry(frame, 8, 1, ui_state, "balancing")
        self.components.options(frame, 8, 2, [str(x) for x in list(BalancingStrategy)], ui_state, "balancing_strategy")

        # loss weight
        self.components.label(frame, 9, 0, "损失权重",
                         tooltip="此数据集的损失乘数")
        self.components.entry(frame, 9, 1, ui_state, "loss_weight")

    def build_image_augmentation_tab(self, frame, controller, image_ui_state):
        # header
        self.components.label(frame, 0, 1, "随机",
                         tooltip="以随机值启用此增强")
        self.components.label(frame, 0, 2, "固定",
                         tooltip="以固定值启用此增强")

        # crop jitter
        self.components.label(frame, 1, 0, "裁剪抖动",
                         tooltip="启用样本随机裁剪")
        self.components.switch(frame, 1, 1, image_ui_state, "enable_crop_jitter")

        # random flip
        self.components.label(frame, 2, 0, "随机翻转",
                         tooltip="训练时随机翻转样本")
        self.components.switch(frame, 2, 1, image_ui_state, "enable_random_flip")
        self.components.switch(frame, 2, 2, image_ui_state, "enable_fixed_flip")

        # random rotation
        self.components.label(frame, 3, 0, "随机旋转",
                         tooltip="训练时随机旋转样本")
        self.components.switch(frame, 3, 1, image_ui_state, "enable_random_rotate")
        self.components.switch(frame, 3, 2, image_ui_state, "enable_fixed_rotate")
        self.components.entry(frame, 3, 3, image_ui_state, "random_rotate_max_angle")

        # random brightness
        self.components.label(frame, 4, 0, "随机亮度",
                         tooltip="训练时随机调整样本亮度")
        self.components.switch(frame, 4, 1, image_ui_state, "enable_random_brightness")
        self.components.switch(frame, 4, 2, image_ui_state, "enable_fixed_brightness")
        self.components.entry(frame, 4, 3, image_ui_state, "random_brightness_max_strength")

        # random contrast
        self.components.label(frame, 5, 0, "随机对比度",
                         tooltip="训练时随机调整样本对比度")
        self.components.switch(frame, 5, 1, image_ui_state, "enable_random_contrast")
        self.components.switch(frame, 5, 2, image_ui_state, "enable_fixed_contrast")
        self.components.entry(frame, 5, 3, image_ui_state, "random_contrast_max_strength")

        # random saturation
        self.components.label(frame, 6, 0, "随机饱和度",
                         tooltip="训练时随机调整样本饱和度")
        self.components.switch(frame, 6, 1, image_ui_state, "enable_random_saturation")
        self.components.switch(frame, 6, 2, image_ui_state, "enable_fixed_saturation")
        self.components.entry(frame, 6, 3, image_ui_state, "random_saturation_max_strength")

        # random hue
        self.components.label(frame, 7, 0, "随机色相",
                         tooltip="训练时随机调整样本色相")
        self.components.switch(frame, 7, 1, image_ui_state, "enable_random_hue")
        self.components.switch(frame, 7, 2, image_ui_state, "enable_fixed_hue")
        self.components.entry(frame, 7, 3, image_ui_state, "random_hue_max_strength")

        # random circular mask shrink
        self.components.label(frame, 8, 0, "圆形遮罩生成",
                         tooltip="自动为遮罩训练创建圆形遮罩")
        self.components.switch(frame, 8, 1, image_ui_state, "enable_random_circular_mask_shrink")

        # random rotate and crop
        self.components.label(frame, 9, 0, "随机旋转裁剪",
                         tooltip="Randomly rotate the training samples and crop to the masked region")
        self.components.switch(frame, 9, 1, image_ui_state, "enable_random_mask_rotate_crop")

        # circular mask generation
        self.components.label(frame, 10, 0, "分辨率覆盖",
                         tooltip="Override the resolution for this concept. Optionally specify multiple resolutions separated by a comma, or a single exact resolution in the format <width>x<height>")
        self.components.switch(frame, 10, 2, image_ui_state, "enable_resolution_override")
        self.components.entry(frame, 10, 3, image_ui_state, "resolution_override")

    def build_text_augmentation_tab(self, frame, controller, text_ui_state):
        # tag shuffling
        self.components.label(frame, 0, 0, "标签打乱",
                         tooltip="启用标签打乱")
        self.components.switch(frame, 0, 1, text_ui_state, "enable_tag_shuffling")

        # keep tag count
        self.components.label(frame, 1, 0, "标签分隔符",
                         tooltip="标签之间的分隔符")
        self.components.entry(frame, 1, 1, text_ui_state, "tag_delimiter")

        # keep tag count
        self.components.label(frame, 2, 0, "保留标签数",
                         tooltip="标签开头不打乱不丢弃的标签数")
        self.components.entry(frame, 2, 1, text_ui_state, "keep_tags_count")

        # tag dropout
        self.components.label(frame, 3, 0, "标签丢弃",
                         tooltip="启用标签随机丢弃")
        self.components.switch(frame, 3, 1, text_ui_state, "tag_dropout_enable")
        self.components.label(frame, 4, 0, "丢弃模式",
                         tooltip="标签丢弃方式：Full整体丢弃，Random随机丢弃，Random Weighted加权丢弃")
        self.components.options_kv(frame, 4, 1, [
            ("全部", 'FULL'),
            ("随机", 'RANDOM'),
            ("随机加权", 'RANDOM WEIGHTED'),
        ], text_ui_state, "tag_dropout_mode", None)
        self.components.label(frame, 4, 2, "概率",
                         tooltip="标签丢弃概率，0到1")
        self.components.entry(frame, 4, 3, text_ui_state, "tag_dropout_probability")

        self.components.label(frame, 5, 0, "特殊丢弃标签",
                         tooltip="丢弃白/黑名单标签列表，可输入分隔列表或文件路径")
        self.components.options_kv(frame, 5, 1, [
            ("无", 'NONE'),
            ("黑名单", 'BLACKLIST'),
            ("白名单", 'WHITELIST'),
        ], text_ui_state, "tag_dropout_special_tags_mode", None)
        self.components.entry(frame, 5, 2, text_ui_state, "tag_dropout_special_tags")
        self.components.label(frame, 6, 0, "特殊标签正则",
                         tooltip="使用正则匹配特殊标签，如'photo.*'匹配'photo, photograph'")
        self.components.switch(frame, 6, 1, text_ui_state, "tag_dropout_special_tags_regex")

        #capitalization randomization
        self.components.label(frame, 7, 0, "随机大小写",
                         tooltip="启用标签大小写随机化")
        self.components.switch(frame, 7, 1, text_ui_state, "caps_randomize_enable")
        self.components.label(frame, 7, 2, "强制小写",
                         tooltip="启用后，将标签转为小写后再处理")
        self.components.switch(frame, 7, 3, text_ui_state, "caps_randomize_lowercase")

        self.components.label(frame, 8, 0, "大小写模式",
                         tooltip="大小写随机化类型：capslock全大写，title首字母大写，first首词大写，random随机")
        self.components.entry(frame, 8, 1, text_ui_state, "caps_randomize_mode")
        self.components.label(frame, 8, 2, "概率",
                         tooltip="Probability to randomize capitialization of each tag, from 0 to 1.")
        self.components.entry(frame, 8, 3, text_ui_state, "caps_randomize_probability")

    def build_concept_stats_tab(self, frame, controller):
        self.concept_stats_tab = frame

        #file size
        self.file_size_label = self.components.label(frame, 1, 0, "总大小", pad=0,
                         tooltip="图像、遮罩和标签文件总大小(MB)", underline=True)
        self.file_size_preview = self.components.label(frame, 2, 0, pad=0, text="-")

        #subdirectory count
        self.dir_count_label = self.components.label(frame, 1, 1, "目录数", pad=0,
                         tooltip="数据集目录及子目录总数", underline=True)
        self.dir_count_preview = self.components.label(frame, 2, 1, pad=0, text="-")

        #basic img/vid stats - count of each type in the concept
        #the \n at the start of the label gives it better vertical spacing with other rows
        self.image_count_label = self.components.label(frame, 3, 0, "\nTotal Images", pad=0,
                         tooltip="图像文件总数，扩展名：" + str(path_util.SUPPORTED_IMAGE_EXTENSIONS) + ", excluding '-masklabel.png and -condlabel.png'", underline=True)
        self.image_count_preview = self.components.label(frame, 4, 0, pad=0, text="-")
        self.video_count_label = self.components.label(frame, 3, 1, "\nTotal Videos", pad=0,
                         tooltip="视频文件总数，扩展名：" + str(path_util.SUPPORTED_VIDEO_EXTENSIONS), underline=True)
        self.video_count_preview = self.components.label(frame, 4, 1, pad=0, text="-")
        self.mask_count_label = self.components.label(frame, 3, 2, "\nTotal Masks", pad=0,
                         tooltip="遮罩文件总数（-masklabel.png结尾）", underline=True)
        self.mask_count_preview = self.components.label(frame, 4, 2, pad=0, text="-")
        self.caption_count_label = self.components.label(frame, 3, 3, "\nTotal Captions", pad=0,
                         tooltip="标签文件总数（.txt文件）", underline=True)
        self.caption_count_preview = self.components.label(frame, 4, 3, pad=0, text="-")

        #advanced img/vid stats - how many img/vid files have a mask or caption of the same name
        self.image_count_mask_label = self.components.label(frame, 5, 0, "\nImages with Masks", pad=0,
                         tooltip="有关联遮罩的图像文件总数", underline=True)
        self.image_count_mask_preview = self.components.label(frame, 6, 0, pad=0, text="-")
        self.mask_count_label_unpaired = self.components.label(frame, 5, 1, "\nUnpaired Masks", pad=0,
                         tooltip="缺少对应图像的遮罩文件数，>0请检查数据集", underline=True)
        self.mask_count_preview_unpaired = self.components.label(frame, 6, 1, pad=0, text="-")
        #currently no masks for videos?

        self.image_count_caption_label = self.components.label(frame, 7, 0, "\nImages with Captions", pad=0,
                         tooltip="有关联标签的图像文件总数", underline=True)
        self.image_count_caption_preview = self.components.label(frame, 8, 0, pad=0, text="-")
        self.video_count_caption_label = self.components.label(frame, 7, 1, "\nVideos with Captions", pad=0,
                         tooltip="有关联标签的视频文件总数", underline=True)
        self.video_count_caption_preview = self.components.label(frame, 8, 1, pad=0, text="-")
        self.caption_count_label_unpaired = self.components.label(frame, 7, 2, "\nUnpaired Captions", pad=0,
                         tooltip="缺少对应图像的标签文件数，>0请检查数据集", underline=True)
        self.caption_count_preview_unpaired = self.components.label(frame, 8, 2, pad=0, text="-")

        #resolution info
        self.pixel_max_label = self.components.label(frame, 9, 0, "\nMax Pixels", pad=0,
                         tooltip="最大图像尺寸（宽x高像素）", underline=True)
        self.pixel_max_preview = self.components.label(frame, 10, 0, pad=0, text="-", wraplength=150)
        self.pixel_avg_label = self.components.label(frame, 9, 1, "\nAvg Pixels", pad=0,
                         tooltip="图像平均尺寸（宽x高像素）", underline=True)
        self.pixel_avg_preview = self.components.label(frame, 10, 1, pad=0, text="-", wraplength=150)
        self.pixel_min_label = self.components.label(frame, 9, 2, "\nMin Pixels", pad=0,
                         tooltip="最小图像尺寸（宽x高像素）", underline=True)
        self.pixel_min_preview = self.components.label(frame, 10, 2, pad=0, text="-", wraplength=150)

        #video length info
        self.length_max_label = self.components.label(frame, 11, 0, "\nMax Length", pad=0,
                         tooltip="数据集中帧数最多的视频", underline=True)
        self.length_max_preview = self.components.label(frame, 12, 0, pad=0, text="-", wraplength=150)
        self.length_avg_label = self.components.label(frame, 11, 1, "\nAvg Length", pad=0,
                         tooltip="视频平均帧数", underline=True)
        self.length_avg_preview = self.components.label(frame, 12, 1, pad=0, text="-", wraplength=150)
        self.length_min_label = self.components.label(frame, 11, 2, "\nMin Length", pad=0,
                         tooltip="数据集中帧数最少的视频", underline=True)
        self.length_min_preview = self.components.label(frame, 12, 2, pad=0, text="-", wraplength=150)

        #video fps info
        self.fps_max_label = self.components.label(frame, 13, 0, "\nMax FPS", pad=0,
                         tooltip="数据集中最高帧率视频", underline=True)
        self.fps_max_preview = self.components.label(frame, 14, 0, pad=0, text="-", wraplength=150)
        self.fps_avg_label = self.components.label(frame, 13, 1, "\nAvg FPS", pad=0,
                         tooltip="数据集中视频平均帧率", underline=True)
        self.fps_avg_preview = self.components.label(frame, 14, 1, pad=0, text="-", wraplength=150)
        self.fps_min_label = self.components.label(frame, 13, 2, "\nMin FPS", pad=0,
                         tooltip="数据集中最低帧率视频", underline=True)
        self.fps_min_preview = self.components.label(frame, 14, 2, pad=0, text="-", wraplength=150)

        #caption info
        self.caption_max_label = self.components.label(frame, 15, 0, "\nMax Caption Length", pad=0,
                         tooltip="最长标签（字符数），Token数约2/词", underline=True)
        self.caption_max_preview = self.components.label(frame, 16, 0, pad=0, text="-", wraplength=150)
        self.caption_avg_label = self.components.label(frame, 15, 1, "\nAvg Caption Length", pad=0,
                         tooltip="标签平均长度（字符数），Token数约2/词", underline=True)
        self.caption_avg_preview = self.components.label(frame, 16, 1, pad=0, text="-", wraplength=150)
        self.caption_min_label = self.components.label(frame, 15, 2, "\nMin Caption Length", pad=0,
                         tooltip="最短标签（字符数），Token数约2/词", underline=True)
        self.caption_min_preview = self.components.label(frame, 16, 2, pad=0, text="-", wraplength=150)

        #aspect bucket info
        self.aspect_bucket_label = self.components.label(frame, 17, 0, "\nAspect Bucketing", pad=0,
                         tooltip="Graph of all possible buckets and the number of images in each one, defined as height/width. Buckets range from 0.25 (4:1 extremely wide) to 4 (1:4 extremely tall). \
                            Images which don't match a bucket exactly are cropped to the nearest one.", underline=True)
        self.small_bucket_label = self.components.label(frame, 17, 1, "\nSmallest Buckets", pad=0,
                         tooltip="非零图像最少的桶，批次大小超过此值时这些图像将被忽略", underline=True)
        self.small_bucket_preview = self.components.label(frame, 18, 1, pad=0, text="-")

        #refresh stats - must be after all labels are defined or will give error
        self.refresh_basic_stats_button = self.components.button(master=frame, row=0, column=0, text="刷新基本", command=lambda: controller.get_concept_stats_threaded(self, False, 9999),
                          tooltip="重新加载数据集目录的基本统计")
        self.refresh_advanced_stats_button = self.components.button(master=frame, row=0, column=1, text="刷新高级", command=lambda: controller.get_concept_stats_threaded(self, True, 9999),
                          tooltip="重新加载数据集目录的高级统计")       #run "basic" scan first before "advanced", seems to help the system cache the directories and run faster
        self.cancel_stats_button = self.components.button(master=frame, row=0, column=2, text="中止扫描", command=lambda: self._cancel_concept_stats(controller),
                          tooltip="如果扫描时间过长则中止——高级扫描对大文件夹和HDD较慢")
        self.processing_time = self.components.label(frame, 0, 3, text="-", tooltip="处理数据集目录耗时")

    def _update_concept_stats(self, controller):
        #file size
        self.components.set_label_text(self.file_size_preview, str(int(controller.concept.concept_stats["file_size"]/1048576)) + " MB")
        self.components.set_label_text(self.processing_time, str(round(controller.concept.concept_stats["processing_time"], 2)) + " s")

        #directory count
        self.components.set_label_text(self.dir_count_preview, controller.concept.concept_stats["directory_count"])

        #image count
        self.components.set_label_text(self.image_count_preview, controller.concept.concept_stats["image_count"])
        self.components.set_label_text(self.image_count_mask_preview, controller.concept.concept_stats["image_with_mask_count"])
        self.components.set_label_text(self.image_count_caption_preview, controller.concept.concept_stats["image_with_caption_count"])

        #video count
        self.components.set_label_text(self.video_count_preview, controller.concept.concept_stats["video_count"])
        #self.components.set_label_text(self.video_count_mask_preview, controller.concept.concept_stats["video_with_mask_count"])
        self.components.set_label_text(self.video_count_caption_preview, controller.concept.concept_stats["video_with_caption_count"])

        #mask count
        self.components.set_label_text(self.mask_count_preview, controller.concept.concept_stats["mask_count"])
        self.components.set_label_text(self.mask_count_preview_unpaired, controller.concept.concept_stats["unpaired_masks"])

        #caption count
        if controller.concept.concept_stats["subcaption_count"] > 0:
            self.components.set_label_text(self.caption_count_preview, f'{controller.concept.concept_stats["caption_count"]} ({controller.concept.concept_stats["subcaption_count"]})')
        else:
            self.components.set_label_text(self.caption_count_preview, controller.concept.concept_stats["caption_count"])
        self.components.set_label_text(self.caption_count_preview_unpaired, controller.concept.concept_stats["unpaired_captions"])

        #resolution info
        max_pixels = controller.concept.concept_stats["max_pixels"]
        avg_pixels = controller.concept.concept_stats["avg_pixels"]
        min_pixels = controller.concept.concept_stats["min_pixels"]

        if any(isinstance(x, str) for x in [max_pixels, avg_pixels, min_pixels]) or controller.concept.concept_stats["image_count"] == 0:   #will be str if adv stats were not taken
            self.components.set_label_text(self.pixel_max_preview, "-")
            self.components.set_label_text(self.pixel_avg_preview, "-")
            self.components.set_label_text(self.pixel_min_preview, "-")
        else:
            #formatted as (#pixels/1000000) MP, width x height, \n filename
            self.components.set_label_text(self.pixel_max_preview, f'{str(round(max_pixels[0]/1000000, 2))} MP, {max_pixels[2]}\n{max_pixels[1]}')
            self.components.set_label_text(self.pixel_avg_preview, f'{str(round(avg_pixels/1000000, 2))} MP, ~{int(math.sqrt(avg_pixels))}w x {int(math.sqrt(avg_pixels))}h')
            self.components.set_label_text(self.pixel_min_preview, f'{str(round(min_pixels[0]/1000000, 2))} MP, {min_pixels[2]}\n{min_pixels[1]}')

        #video length and fps info
        max_length = controller.concept.concept_stats["max_length"]
        avg_length = controller.concept.concept_stats["avg_length"]
        min_length = controller.concept.concept_stats["min_length"]
        max_fps = controller.concept.concept_stats["max_fps"]
        avg_fps = controller.concept.concept_stats["avg_fps"]
        min_fps = controller.concept.concept_stats["min_fps"]

        if any(isinstance(x, str) for x in [max_length, avg_length, min_length]) or controller.concept.concept_stats["video_count"] == 0:   #will be str if adv stats were not taken
            self.components.set_label_text(self.length_max_preview, "-")
            self.components.set_label_text(self.length_avg_preview, "-")
            self.components.set_label_text(self.length_min_preview, "-")
            self.components.set_label_text(self.fps_max_preview, "-")
            self.components.set_label_text(self.fps_avg_preview, "-")
            self.components.set_label_text(self.fps_min_preview, "-")
        else:
            #formatted as (#frames) frames \n filename
            self.components.set_label_text(self.length_max_preview, f'{int(max_length[0])} frames\n{max_length[1]}')
            self.components.set_label_text(self.length_avg_preview, f'{int(avg_length)} frames')
            self.components.set_label_text(self.length_min_preview, f'{int(min_length[0])} frames\n{min_length[1]}')
            #formatted as (#fps) fps \n filename
            self.components.set_label_text(self.fps_max_preview, f'{int(max_fps[0])} fps\n{max_fps[1]}')
            self.components.set_label_text(self.fps_avg_preview, f'{int(avg_fps)} fps')
            self.components.set_label_text(self.fps_min_preview, f'{int(min_fps[0])} fps\n{min_fps[1]}')

        #caption info
        max_caption_length = controller.concept.concept_stats["max_caption_length"]
        avg_caption_length = controller.concept.concept_stats["avg_caption_length"]
        min_caption_length = controller.concept.concept_stats["min_caption_length"]

        if any(isinstance(x, str) for x in [max_caption_length, avg_caption_length, min_caption_length]) or controller.concept.concept_stats["caption_count"] == 0:   #will be str if adv stats were not taken
            self.components.set_label_text(self.caption_max_preview, "-")
            self.components.set_label_text(self.caption_avg_preview, "-")
            self.components.set_label_text(self.caption_min_preview, "-")
        else:
            #formatted as (#chars) chars, (#words) words, \n filename
            self.components.set_label_text(self.caption_max_preview, f'{max_caption_length[0]} chars, {max_caption_length[2]} words\n{max_caption_length[1]}')
            self.components.set_label_text(self.caption_avg_preview, f'{int(avg_caption_length[0])} chars, {int(avg_caption_length[1])} words')
            self.components.set_label_text(self.caption_min_preview, f'{min_caption_length[0]} chars, {min_caption_length[2]} words\n{min_caption_length[1]}')

        #aspect bucketing
        aspect_buckets = controller.concept.concept_stats["aspect_buckets"]
        if len(aspect_buckets) != 0 and max(val for val in aspect_buckets.values()) > 0:    #check aspect_bucket data exists and is not all zero
            min_val = min(val for val in aspect_buckets.values() if val > 0)                #smallest nonzero values
            if max(val for val in aspect_buckets.values()) > min_val:                       #check if any buckets larger than min_val exist - if all images are same aspect then there won't be
                min_val2 = min(val for val in aspect_buckets.values() if (val > 0 and val != min_val))  #second smallest bucket
            else:
                min_val2 = min_val  #if no second smallest bucket exists set to min_val
            min_aspect_buckets = {key: val for key,val in aspect_buckets.items() if val in (min_val, min_val2)}
            min_bucket_str = ""
            for key, val in min_aspect_buckets.items():
                min_bucket_str += f'aspect {self.decimal_to_aspect_ratio(key)} : {val} img\n'
            min_bucket_str.strip()
            self.components.set_label_text(self.small_bucket_preview, min_bucket_str)

        self.bucket_ax.cla()
        aspects = [str(x) for x in list(aspect_buckets.keys())]
        aspect_ratios = [self.decimal_to_aspect_ratio(x) for x in list(aspect_buckets.keys())]
        counts = list(aspect_buckets.values())
        b = self.bucket_ax.bar(aspect_ratios, counts)
        self.bucket_ax.bar_label(b, color=self.text_color)
        sec = self.bucket_ax.secondary_xaxis(location=-0.1)
        sec.spines["bottom"].set_linewidth(0)
        sec.set_xticks([0, (len(aspects)-1)/2, len(aspects)-1], labels=["宽图", "方形", "长图"])
        sec.tick_params('x', length=0)
        self.canvas.draw()

    def decimal_to_aspect_ratio(self, value : float):
        #find closest fraction to decimal aspect value and convert to a:b format
        aspect_fraction = fractions.Fraction(value).limit_denominator(16)
        aspect_string = f'{aspect_fraction.denominator}:{aspect_fraction.numerator}'
        return aspect_string

    def _disable_scan_buttons(self):
        self.components.set_widget_enabled(self.refresh_basic_stats_button, False)
        self.components.set_widget_enabled(self.refresh_advanced_stats_button, False)

    def _enable_scan_buttons(self):
        self.components.set_widget_enabled(self.refresh_basic_stats_button, True)
        self.components.set_widget_enabled(self.refresh_advanced_stats_button, True)

    def _cancel_concept_stats(self, controller):
        controller.cancel_scan_flag.set()
