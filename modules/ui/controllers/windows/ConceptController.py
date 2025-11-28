from modules.ui.controllers.BaseController import BaseController
from modules.ui.models.ConceptModel import ConceptModel
from modules.ui.utils.FigureWidget import FigureWidget
from modules.ui.utils.WorkerPool import WorkerPool
from modules.util.enum.BalancingStrategy import BalancingStrategy
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DropoutMode import DropoutMode
from modules.util.enum.PromptSource import PromptSource
from modules.util.enum.SpecialDropoutTags import SpecialDropoutTags

import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtW
from matplotlib import pyplot as plt
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


class ConceptController(BaseController):
    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/windows/concept.ui", name=None, parent=parent)

    ###FSM###

    def _setup(self):
        self.idx = 0
        self.file_index = 0

        plt.set_loglevel('WARNING')  # suppress errors about data type in bar chart

        self.canvas = FigureWidget(parent=self.ui, width=7, height=3, zoom_tools=True)
        self.bucket_ax = self.canvas.figure.subplots()
        self.ui.histogramLay.addWidget(self.canvas.toolbar) # Matplotlib toolbar, in case we want the user to zoom in.
        self.ui.histogramLay.addWidget(self.canvas)

    def _connectUIBehavior(self):
        self._connectFileDialog(self.ui.pathBtn, self.ui.pathLed, is_dir=True, title=QCA.translate("dialog_window", "Open Dataset directory"))
        self._connectFileDialog(self.ui.promptSourceBtn, self.ui.promptSourceLed, is_dir=False,
                               title=QCA.translate("dialog_window", "Open Prompt Source"),
                               filters=QCA.translate("filetype_filters",
                                                     "Text (*.txt)"))

        self._connect(QtW.QApplication.instance().openConcept, self.__updateConcept())
        self._connect(QtW.QApplication.instance().openConcept, self.__updateStats())
        self._connect(self.ui.okBtn.clicked, self.__saveConcept())


        self._connect(QtW.QApplication.instance().openConcept, self.__updateImage())

        self.dynamic_state_ui_connections = {
            # General tab.
            "{idx}.enabled": "enabledCbx",
            "{idx}.type": "conceptTypeCmb",
            "{idx}.path": "pathLed",
            "{idx}.text.prompt_source": "promptSourceCmb",
            "{idx}.text.prompt_path": "promptSourceLed",
            "{idx}.include_subdirectories": "includeSubdirectoriesCbx",
            "{idx}.image_variations": "imageVariationsSbx",
            "{idx}.text_variations": "textVariationsSbx",
            "{idx}.balancing": "balancingSbx",
            "{idx}.balancing_strategy": "balancingCmb",
            "{idx}.loss_weight": "lossWeightSbx",
            # Image augmentation tab.
            "{idx}.image.enable_crop_jitter": "rndJitterCbx",
            "{idx}.image.enable_random_flip": "rndFlipCbx",
            "{idx}.image.enable_fixed_flip": "fixFlipCbx",
            "{idx}.image.enable_random_rotate": "rndRotationCbx",
            "{idx}.image.enable_fixed_rotate": "fixRotationCbx",
            "{idx}.image.random_rotate_max_angle": "rotationSbx",
            "{idx}.image.enable_random_brightness": "rndBrightnessCbx",
            "{idx}.image.enable_fixed_brightness": "fixBrightnessCbx",
            "{idx}.image.random_brightness_max_strength": "brightnessSbx",
            "{idx}.image.enable_random_contrast": "rndContrastCbx",
            "{idx}.image.enable_fixed_contrast": "fixContrastCbx",
            "{idx}.image.random_contrast_max_strength": "contrastSbx",
            "{idx}.image.enable_random_saturation": "rndSaturationCbx",
            "{idx}.image.enable_fixed_saturation": "fixSaturationCbx",
            "{idx}.image.random_saturation_max_strength": "saturationSbx",
            "{idx}.image.enable_random_hue": "rndHueCbx",
            "{idx}.image.enable_fixed_hue": "fixHueCbx",
            "{idx}.image.random_hue_max_strength": "hueSbx",
            "{idx}.image.enable_resolution_override": "fixResolutionOverrideCbx",
            "{idx}.image.resolution_override": "resolutionOverrideLed",
            "{idx}.image.enable_random_circular_mask_shrink": "rndCircularMaskCbx",
            "{idx}.image.enable_random_mask_rotate_crop": "rndRotateCropCbx",
            # Text augmentation tab.
            "{idx}.text.enable_tag_shuffling": "tagShufflingCbx",
            "{idx}.text.tag_delimiter": "tagDelimiterLed",
            "{idx}.text.keep_tags_count": "keepTagCountSbx",
            "{idx}.text.tag_dropout_enable": "tagDropoutCbx",
            "{idx}.text.tag_dropout_mode": "dropoutModeCmb",
            "{idx}.text.tag_dropout_probability": "dropoutProbabilitySbx",
            "{idx}.text.tag_dropout_special_tags_mode": "specialDropoutTagsCmb",
            "{idx}.text.tag_dropout_special_tags": "specialDropoutTagsLed",
            "{idx}.text.tag_dropout_special_tags_regex": "specialTagsRegexCbx",
            "{idx}.text.caps_randomize_enable": "randomizeCapitalizationCbx",
            "{idx}.text.caps_randomize_probability": "capitalizationProbabilitySbx",
            "{idx}.text.caps_randomize_mode": "capitalizationModeLed",
            "{idx}.text.caps_randomize_lowercase": "forceLowercaseCbx",
        }

        self._connect(QtW.QApplication.instance().openConcept, self.__reconnectControls())

        self._connect(self.ui.promptSourceCmb.activated, self.__enablePromptSource())
        self._connect(self.ui.refreshBasicBtn.clicked, self.__startScan(advanced_scanning=False))
        self._connect(self.ui.refreshAdvancedBtn.clicked, self.__startScan(advanced_scanning=True))
        self._connect(self.ui.abortScanBtn.clicked, self.__abortScan())
        self._connect(self.ui.downloadNowBtn.clicked, self.__startDownload())
        self._connect(self.ui.prevBtn.clicked, self.__prevImage())
        self._connect(self.ui.nextBtn.clicked, self.__nextImage())

        self._connect([self.ui.prevBtn.clicked, self.ui.nextBtn.clicked, self.ui.updatePreviewBtn.clicked],
                      self.__updateImage())

        self.__enableDownloadBtn(True)()
        self.__enableScanBtn(True)()


    def _loadPresets(self):
        for e in PromptSource.enabled_values():
            self.ui.promptSourceCmb.addItem(e.pretty_print(), userData=str(e)) # ConceptConfig serializes string, not enum

        for e in DropoutMode.enabled_values():
            self.ui.dropoutModeCmb.addItem(e.pretty_print(), userData=str(e)) # ConceptConfig serializes string, not enum

        for e in SpecialDropoutTags.enabled_values():
            self.ui.specialDropoutTagsCmb.addItem(e.pretty_print(), userData=str(e)) # ConceptConfig serializes string, not enum

        for e in BalancingStrategy.enabled_values():
            self.ui.balancingCmb.addItem(e.pretty_print(), userData=e)

        # This always allows Prior Validation concepts, even when LORA is not selected. (The behavior is the same as original OneTrainer, delegating checks to non-ui methods).
        for e in ConceptType.enabled_values(context="prior_pred_enabled"):
            self.ui.conceptTypeCmb.addItem(e.pretty_print(), userData=e)

    def _connectInputValidation(self):
        self.ui.resolutionOverrideLed.setValidator(QtGui.QRegularExpressionValidator(r"\d+(x\d+(,\d+x\d+)*)?", self.ui))

    ###Reactions###

    def __reconnectControls(self):
        @Slot()
        def f():
            self._disconnectGroup("idx")
            self._connectStateUI(self.dynamic_state_ui_connections, ConceptModel.instance(), signal=None, group="idx", update_after_connect=True, idx=self.idx)
        return f

    def __updateStats(self):
        @Slot()
        def f():
            self.__enableScanBtn(True)()
            stats_dict = ConceptModel.instance().pretty_print_stats(self.idx)

            for k, v in {
                "fileSizeLbl": "file_size",
                "processingTimeLbl": "processing_time",
                "dirCountLbl": "dir_count",
                "imageCountLbl": "image_count",
                "imageCountMaskLbl": "image_count_mask",
                "imageCountCaptionLbl": "image_count_caption",
                "videoCountLbl": "video_count",
                "videoCountCaptionLbl": "video_count_caption",
                "maskCountLbl": "mask_count",
                "maskCountUnpairedLbl": "mask_count_unpaired",
                "captionCountLbl": "caption_count",
                "unpairedCaptionsLbl": "unpaired_captions",
                "maxPixelsLbl": "max_pixels",
                "avgPixelsLbl": "avg_pixels",
                "minPixelsLbl": "min_pixels",
                "lengthMaxLbl": "length_max",
                "lengthAvgLbl": "length_avg",
                "lengthMinLbl": "length_min",
                "fpsMaxLbl": "fps_max",
                "fpsAvgLbl": "fps_avg",
                "fpsMinLbl": "fps_min",
                "captionMaxLbl": "caption_max",
                "captionAvgLbl": "caption_avg",
                "captionMinLbl": "caption_min",
                "smallBucketLbl": "small_bucket",
            }.items():
                self.ui.findChild(QtW.QLabel, k).setText(str(stats_dict[v]))

            self.__updateHistogram(stats_dict)
        return f

    def __startScan(self, advanced_scanning):
        @Slot()
        def f():
            worker, name = WorkerPool.instance().createNamed(self.__scanConcept(), "scan_concept", abort_flag=ConceptModel.instance().cancel_scan_flag, advanced_scanning=advanced_scanning)
            if worker is not None:
                worker.connectCallbacks(init_fn=self.__enableScanBtn(False), result_fn=None, finished_fn=self.__updateStats(), errored_fn=self.__enableScanBtn(True), aborted_fn=self.__enableScanBtn(True))
                WorkerPool.instance().start(name)

        return f

    def __startDownload(self):
        @Slot()
        def f():
            worker, name = WorkerPool.instance().createNamed(self.__downloadConcept(), "download_concept")
            if worker is not None:
                worker.connectCallbacks(init_fn=self.__enableDownloadBtn(False), result_fn=None, finished_fn=self.__enableDownloadBtn(True),
                               errored_fn=self.__enableDownloadBtn(True), aborted_fn=self.__enableDownloadBtn(True))
                WorkerPool.instance().start(name)
        return f

    def __downloadConcept(self):
        @Slot()
        def f():
            ConceptModel.instance().download_dataset(self.idx)
        return f

    def __abortScan(self):
        @Slot()
        def f():
            ConceptModel.instance().cancel_scan_flag.set()
        return f

    def __enablePromptSource(self):
        @Slot(int)
        def f(value):
            if self.ui.promptSourceCmb.currentData() != "concept": # TODO: Replace with "PromptSource.CONCEPT" when ConceptConfig will accept enum instead of string.
                self.ui.promptSourceLed.setEnabled(False)
                self.ui.promptSourceBtn.setEnabled(False)
            else:
                self.ui.promptSourceLed.setEnabled(True)
                self.ui.promptSourceBtn.setEnabled(True)
        return f

    def __prevImage(self):
        @Slot()
        def f():
            image_count = ConceptModel.instance().get_state(f"{self.idx}.concept_stats.image_count")
            if image_count is not None and image_count > 0:
                self.file_index = (self.file_index + image_count - 1) % image_count
            else:
                self.file_index = max(0, self.file_index - 1)
        return f

    def __nextImage(self):
        @Slot()
        def f():
            image_count = ConceptModel.instance().get_state(f"{self.idx}.concept_stats.image_count")
            if image_count is not None and image_count > 0:
                self.file_index = (self.file_index + 1) % image_count
            else:
                self.file_index += 1
        return f

    def __updateImage(self):
        @Slot()
        def f():
            img, filename, caption = ConceptModel.instance().getImage(self.idx, self.file_index, show_augmentations=self.ui.showAugmentationsCbx.isChecked())
            self.ui.previewLbl.setPixmap(QtGui.QPixmap.fromImage(ImageQt(img)))
            self.ui.filenameLbl.setText(filename)
            self.ui.promptTed.setPlainText(caption)
        return f

    def __updateConcept(self):
        @Slot(int)
        def f(idx):
            self.idx = idx
            self.file_index = 0

            self.ui.nameLed.setText(ConceptModel.instance().get_concept_name(self.idx)) # Name has a different logic than other controls and cannot exploit the connection dictionary.
        return f

    def __saveConcept(self):
        @Slot()
        def f():
            ConceptModel.instance().set_state(f"{self.idx}.name", self.ui.nameLed.text())

            # No need to store statistics, as they are handled directly by the model.
            QtW.QApplication.instance().conceptsChanged.emit(True)
            self.ui.hide()
        return f

    def __enableDownloadBtn(self, enabled):
        @Slot()
        def f():
            self.ui.downloadNowBtn.setEnabled(enabled)
        return f

    def __enableScanBtn(self, enabled):
        @Slot()
        def f():
            self.ui.refreshBasicBtn.setEnabled(enabled)
            self.ui.refreshAdvancedBtn.setEnabled(enabled)
            self.ui.abortScanBtn.setEnabled(not enabled)
        return f

    ###Utils###

    def __updateHistogram(self, stats_dict):
        self.bucket_ax.cla()
        self.canvas.figure.tight_layout()
        self.canvas.figure.subplots_adjust(bottom=0.15)
        self.bucket_ax.spines['top'].set_visible(False)
        self.bucket_ax.tick_params(axis='x', which="both")
        self.bucket_ax.tick_params(axis='y', which="both")
        aspects = [str(x) for x in list(stats_dict["aspect_buckets"].keys())]
        aspect_ratios = [ConceptModel.instance().decimal_to_aspect_ratio(x) for x in
                         list(stats_dict["aspect_buckets"].keys())]
        counts = list(stats_dict["aspect_buckets"].values())
        b = self.bucket_ax.bar(aspect_ratios, counts)
        self.bucket_ax.bar_label(b)
        sec = self.bucket_ax.secondary_xaxis(location=-0.1)
        sec.spines["bottom"].set_linewidth(0)
        sec.set_xticks([0, (len(aspects) - 1) / 2, len(aspects) - 1], labels=["Wide", "Square", "Tall"])
        sec.tick_params('x', length=0)
        self.canvas.draw_idle()

    def __scanConcept(self):
        def f(advanced_scanning):
            return ConceptModel.instance().get_concept_stats(self.idx, advanced_scanning)
        return f
