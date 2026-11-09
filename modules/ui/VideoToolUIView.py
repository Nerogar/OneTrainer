import webbrowser

from modules.ui.VideoToolUIController import VideoToolUIController
from modules.util.image_util import load_image
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QWidget,
)

_PAD = pyside6_components.PAD


class VideoToolUIView(QDialog):
    def __init__(self, parent, controller: VideoToolUIController):
        QDialog.__init__(self, parent)
        self.components = pyside6_components

        self.controller = controller
        self._status_box: QTextEdit | None = None
        self._preview_label: QLabel | None = None
        self._preview_caption_label: QLabel | None = None

        ui_state = PySide6UIState(controller.args)

        self.setWindowTitle("Video Tools")
        self.resize(700, 750)

        outer = QGridLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.setRowStretch(0, 1)
        outer.setRowStretch(1, 0)

        tabs = QTabWidget(self)
        outer.addWidget(tabs, 0, 0)

        for name, build_fn in [
            ("extract clips", self.build_clip_extract_tab),
            ("extract images", self.build_image_extract_tab),
            ("download", self.build_video_download_tab),
        ]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            frame = QWidget()
            scroll.setWidget(frame)
            lo = pyside6_components._layout(frame)
            lo.setContentsMargins(_PAD, _PAD, _PAD, _PAD)
            lo.setColumnMinimumWidth(0, 120)
            lo.setColumnStretch(3, 1)
            build_fn(frame, controller, ui_state)
            lo.setRowStretch(lo.rowCount(), 1)
            tabs.addTab(scroll, name)

        outer.addWidget(self._build_status_bar(), 1, 0)

    def _build_status_bar(self):
        frame = QWidget(self)
        lo = QGridLayout(frame)
        lo.setColumnMinimumWidth(0, 160)
        lo.setColumnStretch(2, 1)

        self._preview_label = QLabel(frame)
        self._preview_label.setFixedSize(150, 150)
        preview = load_image("resources/icons/icon.png", 'RGB')
        preview.thumbnail((150, 150))
        self._preview_label.setPixmap(
            QPixmap.fromImage(ImageQt(preview.convert("RGBA"))).scaled(
                150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        self._preview_caption_label = QLabel("Preview image", frame)
        self._preview_caption_label.setWordWrap(True)

        preview_col = QWidget(frame)
        preview_lo = QGridLayout(preview_col)
        preview_lo.setContentsMargins(0, 0, 0, 0)
        preview_lo.addWidget(self._preview_label, 0, 0, Qt.AlignTop)
        preview_lo.addWidget(self._preview_caption_label, 1, 0, Qt.AlignTop)
        lo.addWidget(preview_col, 0, 0, Qt.AlignTop | Qt.AlignLeft)

        self._status_box = QTextEdit(frame)
        self._status_box.setReadOnly(True)
        self._status_box.setFixedHeight(160)
        self._status_box.setMinimumWidth(300)
        self._status_box.setPlainText("Current status")
        lo.addWidget(self._status_box, 0, 1, Qt.AlignTop)

        return frame

    # --- abstract method implementations ---

    def _create_textbox(self, master, row, col, width, height, ui_state, var_name):
        var = ui_state.get_var(var_name)
        widget = QTextEdit(master)
        widget.setFixedHeight(height)
        widget.setMinimumWidth(width)
        widget.setPlainText(var.get())
        pyside6_components._add(
            pyside6_components._layout(master), widget, row, col, sticky="w", rowspan=2
        )
        widget.textChanged.connect(lambda: var.set(widget.toPlainText()))
        return widget

    def schedule_on_main_thread(self, fn):
        QTimer.singleShot(0, self, fn)

    def update_status(self, status_text: str):
        # Called from the video tool's worker thread — defer to main thread
        self.schedule_on_main_thread(lambda: self._status_box.append(status_text))

    def clear_status(self):
        self._status_box.clear()

    def update_preview(self, preview_image, label_text: str):
        # Called from the video tool's worker thread — defer to main thread
        pixmap = QPixmap.fromImage(ImageQt(preview_image.convert("RGBA")))
        self.schedule_on_main_thread(lambda: self._do_update_preview(pixmap, label_text))

    def _do_update_preview(self, pixmap: QPixmap, label_text: str):
        self._preview_label.setPixmap(
            pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self._preview_caption_label.setText(label_text)

    def build_clip_extract_tab(self, frame, controller, ui_state):
        # single video
        self.components.label(frame, 0, 0, "Single Video",
                         tooltip="Link to single video file to process.")
        self.components.path_entry(frame, 0, 1, ui_state, "clip_single",
                               mode="file", allow_model_files=False, allow_video_files=True)
        self.components.button(frame, 0, 2, "Extract Single",
                          command=lambda: self._extract_clips(False, controller))

        # time range
        self.components.label(frame, 1, 0, "  Time Range",
                         tooltip="Time range to limit selection for single video, \
                            format as hour:minute:second, minute:second, or seconds.")
        self.components.entry(frame, 1, 1, ui_state, "clip_time_start", width=100, sticky="w")
        self.components.entry(frame, 1, 1, ui_state, "clip_time_end", width=100, sticky="e")

        # directory of videos
        self.components.label(frame, 2, 0, "Directory",
                         tooltip="Path to directory with multiple videos to process, including in subdirectories.")
        self.components.path_entry(frame, 2, 1, ui_state, "clip_list", mode="dir")
        self.components.button(frame, 2, 2, "Extract Directory",
                          command=lambda: self._extract_clips(True, controller))

        # output directory
        self.components.label(frame, 3, 0, "Output",
                         tooltip="Path to folder where extracted clips will be saved.")
        self.components.path_entry(frame, 3, 1, ui_state, "clip_output", mode="dir")

        # output to subdirectories
        self.components.label(frame, 4, 0, "Output to\nSubdirectories",
                         tooltip="If enabled, files are saved to subfolders based on filename and input directory. \
                            Otherwise will all be saved to the top level of the output directory.")
        self.components.switch(frame, 4, 1, ui_state, "output_subdir_clip")

        # split at cuts
        self.components.label(frame, 5, 0, "Split at Cuts",
                         tooltip="If enabled, detect cuts in the input video and split at those points. \
                            Otherwise will split at any point, and clips may contain cuts.")
        self.components.switch(frame, 5, 1, ui_state, "split_cuts")

        # maximum length
        self.components.label(frame, 6, 0, "Max Length (s)",
                         tooltip="Maximum length in seconds for saved clips, larger clips will be broken into multiple small clips.")
        self.components.entry(frame, 6, 1, ui_state, "clip_length", width=220)

        # Set FPS
        self.components.label(frame, 7, 0, "Set FPS",
                         tooltip="FPS to convert output videos to, set to 0 to keep original rate.")
        self.components.entry(frame, 7, 1, ui_state, "clip_fps", width=220)

        # Remove borders
        self.components.label(frame, 8, 0, "Remove Borders",
                         tooltip="Remove black borders from output clip")
        self.components.switch(frame, 8, 1, ui_state, "clip_bordercrop")

        # Crop Variation
        self.components.label(frame, 9, 0, "Crop Variation",
                         tooltip="Output clips will be randomly cropped to +- the base aspect ratio, \
                              somewhat biased towards making square videos. Set to 0 to use only base aspect.")
        self.components.entry(frame, 9, 1, ui_state, "clip_crop", width=220)

    def build_image_extract_tab(self, frame, controller, ui_state):
        # single video
        self.components.label(frame, 0, 0, "Single Video",
                         tooltip="Link to single video file to process.")
        self.components.path_entry(frame, 0, 1, ui_state, "image_single",
                               mode="file", allow_model_files=False, allow_video_files=True)
        self.components.button(frame, 0, 2, "Extract Single",
                          command=lambda: self._extract_images(False, controller))

        # time range
        self.components.label(frame, 1, 0, "  Time Range",
                         tooltip="Time range to limit selection for single video, \
                            format as hour:minute:second, minute:second, or seconds.")
        self.components.entry(frame, 1, 1, ui_state, "image_time_start", width=100, sticky="w")
        self.components.entry(frame, 1, 1, ui_state, "image_time_end", width=100, sticky="e")

        # directory of videos
        self.components.label(frame, 2, 0, "Directory",
                         tooltip="Path to directory with multiple videos to process, including in subdirectories.")
        self.components.path_entry(frame, 2, 1, ui_state, "image_list", mode="dir")
        self.components.button(frame, 2, 2, "Extract Directory",
                          command=lambda: self._extract_images(True, controller))

        # output directory
        self.components.label(frame, 3, 0, "Output",
                         tooltip="Path to folder where extracted images will be saved.")
        self.components.path_entry(frame, 3, 1, ui_state, "image_output", mode="dir")

        # output to subdirectories
        self.components.label(frame, 4, 0, "Output to\nSubdirectories",
                         tooltip="If enabled, files are saved to subfolders based on filename and input directory. \
                            Otherwise will all be saved to the top level of the output directory.")
        self.components.switch(frame, 4, 1, ui_state, "output_subdir_img")

        # image capture rate
        self.components.label(frame, 5, 0, "Images/sec",
                         tooltip="Number of images to capture per second of video. \
                            Images will be taken at semi-random frames around the specified frequency.")
        self.components.entry(frame, 5, 1, ui_state, "capture_rate", width=220)

        # blur removal
        self.components.label(frame, 6, 0, "Blur Removal",
                         tooltip="Threshold for removal of blurry images, relative to all others. \
                            For example at 0.2, the blurriest 20%% of the final selected frames will not be saved.")
        self.components.entry(frame, 6, 1, ui_state, "blur_threshold", width=220)

        # Remove borders
        self.components.label(frame, 7, 0, "Remove Borders",
                         tooltip="Remove black borders from output image")
        self.components.switch(frame, 7, 1, ui_state, "image_bordercrop")

        # Crop Variation
        self.components.label(frame, 8, 0, "Crop Variation",
                         tooltip="Output images will be randomly cropped to +- the base aspect ratio, \
                            somewhat biased towards making square images. Set to 0 to use only base sapect.")
        self.components.entry(frame, 8, 1, ui_state, "image_crop", width=220)

    def build_video_download_tab(self, frame, controller, ui_state):
        # link
        self.components.label(frame, 0, 0, "Single Link",
                         tooltip="Link to video/playlist to download. Uses yt-dlp, supports youtube, twitch, instagram, and many other sites.")
        self.components.entry(frame, 0, 1, ui_state, "download_link", width=220)
        self.components.button(frame, 0, 2, "Download Link",
                          command=lambda: self._download(False, controller))

        # link list
        self.components.label(frame, 1, 0, "Link List",
                         tooltip="Path to txt file with list of links separated by newlines.")
        self.components.path_entry(frame, 1, 1, ui_state, "download_list",
                               mode="file", allow_model_files=False)
        self.components.button(frame, 1, 2, "Download List",
                          command=lambda: self._download(True, controller))

        # output directory
        self.components.label(frame, 2, 0, "Output",
                         tooltip="Path to folder where downloaded videos will be saved.")
        self.components.path_entry(frame, 2, 1, ui_state, "download_output", mode="dir")

        # additional args
        self.components.label(frame, 3, 0, "Additional Args",
                         tooltip="Any additional arguments to pass to yt-dlp, for example '--restrict-filenames --force-overwrite'. \
                            Default args will hide most terminal outputs.")
        self._create_textbox(frame, 3, 1, 220, 90, ui_state, "download_args")
        self.components.button(frame, 3, 2, "yt-dlp info",
                          command=lambda: webbrowser.open("https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#usage-and-options", new=0, autoraise=False))

    def _extract_clips(self, batch_mode: bool, controller):
        controller.extract_clips_button(batch_mode)

    def _extract_images(self, batch_mode: bool, controller):
        controller.extract_images_button(batch_mode)

    def _download(self, batch_mode: bool, controller):
        controller.download_button(batch_mode)
