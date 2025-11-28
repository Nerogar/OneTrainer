from modules.ui.controllers.BaseController import BaseController
from modules.ui.models.DatasetModel import DatasetModel
from modules.ui.models.MaskHistoryModel import MaskHistoryModel
from modules.ui.utils.FigureWidget import FigureWidget
from modules.ui.utils.WorkerPool import WorkerPool
from modules.util.enum.CaptionFilter import CaptionFilter
from modules.util.enum.EditMode import EditMode
from modules.util.enum.FileFilter import FileFilter
from modules.util.enum.MouseButton import MouseButton
from modules.util.enum.ToolType import ToolType

import numpy as np
import PySide6.QtGui as QtG
import PySide6.QtWidgets as QtW
from matplotlib.transforms import Bbox
from PIL import Image
from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


class DatasetController(BaseController):
    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/windows/dataset.ui", name=None, parent=parent)

    ###FSM###

    def _setup(self):
        self.theme = "dark" if QtG.QGuiApplication.styleHints().colorScheme() == QtG.Qt.ColorScheme.Dark else "light"

        # (fn, type, text, name, icon, tooltip, shortcut, spinbox_range)
        # spinbox_range = (min, step, max, default value)
        self.tools = [
            {
                "fn": self.__prevImg(),
                "type": ToolType.BUTTON,
                "icon": f"resources/icons/buttons/{self.theme}/arrow-left.svg",
                "tooltip": QCA.translate("toolbar_item", "Previous image (Left Arrow)"),
                "shortcut": "Left"
            },
            {
                "fn": self.__nextImg(),
                "type": ToolType.BUTTON,
                "icon": f"resources/icons/buttons/{self.theme}/arrow-right.svg",
                "tooltip": QCA.translate("toolbar_item", "Next image (Right Arrow)"),
                "shortcut": "Right"
            },
            {"type": ToolType.SEPARATOR},
            {
                "type": ToolType.CHECKABLE_BUTTON,
                "tool": EditMode.DRAW,
                "icon": f"resources/icons/buttons/{self.theme}/brush.svg",
                "tooltip": QCA.translate("toolbar_item", "Draw (Left Click) or Erase (Right Click) mask (CTRL+E)"),
                "shortcut": "Ctrl+E"
            },
            {
                "type": ToolType.CHECKABLE_BUTTON,
                "tool": EditMode.FILL,
                "icon": f"resources/icons/buttons/{self.theme}/paint-bucket.svg",
                "tooltip": QCA.translate("toolbar_item", "Fill (Left Click) or Erase-fill (Right Click) mask (CTRL+F)"),
                "shortcut": "Ctrl+F"
            },
            {
                "fn": self.__setBrushSize(),
                "type": ToolType.SPINBOX,
                "name": "brush_sbx",
                "text": QCA.translate("toolbar_item", "Brush size"),
                "tooltip": QCA.translate("toolbar_item", "Brush size (Mouse Wheel Up/Down)"),
                "spinbox_range": (1, 256, 1),
                "value": 10
            },
            {
                "fn": self.__setAlpha(),
                "type": ToolType.DOUBLE_SPINBOX,
                "name": "alpha_sbx",
                "text": QCA.translate("toolbar_item", "Mask opacity"),
                "tooltip": QCA.translate("toolbar_item", "Mask opacity for preview"),
                "spinbox_range": (0.05, 1.0, 0.05),
                "value": 0.5
            },
            {"type": ToolType.SEPARATOR},
            {
                "fn": self.__clearMask(),
                "type": ToolType.BUTTON,
                "text": QCA.translate("toolbar_item", "Clear Mask"),
                "tooltip": QCA.translate("toolbar_item", "Clear mask (Del, or Middle Click)"),
                "shortcut": "Del"
            },
            {
                "fn": self.__clearAll(),
                "type": ToolType.BUTTON,
                "text": QCA.translate("toolbar_item", "Clear All"),
                "tooltip": QCA.translate("toolbar_item", "Clear mask and caption (CTRL+Del)"),
                "shortcut": "Ctrl+Del"
            },
            {
                "fn": self.__resetMask(),
                "type": ToolType.BUTTON,
                "text": QCA.translate("toolbar_item", "Reset Mask"),
                "tooltip": QCA.translate("toolbar_item", "Reset mask (CTRL+R)"),
                "shortcut": "Ctrl+R"
            },
            {"type": ToolType.SEPARATOR},
            {
                "fn": self.__saveMask(),
                "type": ToolType.BUTTON,
                "icon": f"resources/icons/buttons/{self.theme}/save.svg",
                "tooltip": QCA.translate("toolbar_item", "Save mask (CTRL+S)"),
                "shortcut": "Ctrl+S"
            },
            {
                "fn": self.__undo(),
                "type": ToolType.BUTTON,
                "icon": f"resources/icons/buttons/{self.theme}/undo.svg",
                "tooltip": QCA.translate("toolbar_item", "Undo (CTRL+Z)"),
                "shortcut": "Ctrl+Z"
            },
            {
                "fn": self.__redo(),
                "type": ToolType.BUTTON,
                "icon": f"resources/icons/buttons/{self.theme}/redo.svg",
                "tooltip": QCA.translate("toolbar_item", "Redo (CTRL+Y)"),
                "shortcut": "Ctrl+Y"
            },
            {"type": ToolType.SEPARATOR},
            {
                "fn": self.__deleteSample(),
                "type": ToolType.BUTTON,
                "icon": f"resources/icons/buttons/{self.theme}/trash-2.svg",
                "tooltip": QCA.translate("toolbar_item", "Delete image, mask and caption (CTRL+SHIFT+Del)"),
                "shortcut": "Ctrl+Shift+Del"
            },
        ]

        self.num_files = 0
        self.current_index = 0
        self.alpha = 1.0
        self.brush = 1
        self.im = None
        self.image = None
        self.current_image_path = None

        self.canvas = FigureWidget(parent=self.ui, width=7, height=5, zoom_tools=True, other_tools=self.tools, emit_clicked=True, emit_moved=True, emit_wheel=True, emit_released=True, use_data_coordinates=True)
        self.ax = self.canvas.figure.subplots() # TODO: when panning, the drawing area changes size. Probably there is some matplotlib option to set.
        self.ax.set_axis_off()

        self.ui.canvasLay.addWidget(self.canvas.toolbar)
        self.ui.canvasLay.addWidget(self.canvas)

        self.leafWidgets = {}

    def _connectUIBehavior(self):
        state_ui_connections = {
            "include_subdirectories": "includeSubdirCbx",
            "file_filter": "fileFilterLed",
            "file_filter_mode": "fileFilterCmb",
            "caption_filter": "captionFilterLed",
            "caption_filter_mode": "captionFilterCmb",
        }

        self._connectStateUI(state_ui_connections, DatasetModel.instance(), signal=None, update_after_connect=True)

        self._connect(self.ui.openBtn.clicked, self.__openDataset())
        self._connect(self.ui.browseBtn.clicked, self.__browse())

        self._connect(self.ui.saveCaptionBtn.clicked, self.__saveCaption())
        self._connect(self.ui.deleteCaptionBtn.clicked, self.__deleteCaption())
        self._connect(self.ui.resetCaptionBtn.clicked, self.__resetCaption())

        self._connect(self.ui.fileTreeWdg.itemSelectionChanged, self.__selectFile())

        self._connect(self.ui.fileFilterLed.editingFinished, self.__updateDataset())
        self._connect(self.ui.fileFilterCmb.activated, self.__updateDataset())
        self._connect(self.ui.captionFilterLed.editingFinished, self.__updateDataset())
        self._connect(self.ui.captionFilterCmb.activated, self.__updateDataset())


        self._connect(self.canvas.clicked, self.__onClicked())
        self._connect(self.canvas.released, self.__onReleased())
        self._connect(self.canvas.wheelUp, self.__onWheelUp())
        self._connect(self.canvas.wheelDown, self.__onWheelDown())

        self.canvas.registerTool(EditMode.DRAW, moved_fn=self.__onDrawMoved(), use_mpl_event=False)
        self.canvas.registerTool(EditMode.FILL, clicked_fn=self.__onMaskClicked(), use_mpl_event=False)


    def _loadPresets(self):
        for e in FileFilter.enabled_values():
            self.ui.fileFilterCmb.addItem(e.pretty_print(), userData=e)

        for e in CaptionFilter.enabled_values():
            self.ui.captionFilterCmb.addItem(e.pretty_print(), userData=e)

    ###Reactions###

    def __openDataset(self):
        @Slot()
        def f():
            diag = QtW.QFileDialog()
            dir = diag.getExistingDirectory(parent=None, caption=QCA.translate("dialog_window", "Open Dataset directory"), dir=DatasetModel.instance().get_state("path"))

            worker, name = WorkerPool.instance().createNamed(self.__scan(), name="open_dataset", dir=dir)
            if worker is not None:
                worker.connectCallbacks(finished_fn=self.__updateDataset())
                WorkerPool.instance().start(name)

        return f

    def __updateDataset(self):
        @Slot()
        def f():
            files = DatasetModel.instance().getFilteredFiles()
            self.current_index = 0
            self.num_files = len(files)
            if self.num_files == 0:
                self.ui.numFilesLbl.setText(QCA.translate("dataset_window", "No image found"))
            else:
                self.ui.numFilesLbl.setText(QCA.translate("dataset_window", "Dataset loaded"))

            file_tree = {}
            for i, file in enumerate(files):
                self.__buildTree(file, file_tree, i)

            self.ui.fileTreeWdg.clear()
            self.leafWidgets = {}
            self.__drawTree(self.ui.fileTreeWdg, file_tree)
        return f


    def __prevImg(self):
        @Slot()
        def f():
            if self.num_files > 0 and self.__saveChanged():
                    self.current_index = (self.current_index + self.num_files - 1) % self.num_files
                    self.ui.fileTreeWdg.setCurrentItem(self.leafWidgets[self.current_index])
        return f

    def __nextImg(self):
        @Slot()
        def f():
            if self.num_files > 0 and self.__saveChanged():
                self.current_index = (self.current_index + 1) % self.num_files
                self.ui.fileTreeWdg.setCurrentItem(self.leafWidgets[self.current_index])
        return f

    def __setBrushSize(self):
        @Slot(int)
        def f(val):
            self.brush = val
        return f

    def __setAlpha(self):
        @Slot(float)
        def f(val):
            self.alpha = val

            self.__updateCanvas()
        return f

    def __clearAll(self):
        @Slot()
        def f():
            choice = self._openAlert(QCA.translate("dataset_window", "Clear Mask and Caption"),
                                    QCA.translate("dataset_window",
                                                  "Do you want to clear mask and caption? This operation will not change files on disk."),
                                    type="question",
                                    buttons=QtW.QMessageBox.StandardButton.Yes | QtW.QMessageBox.StandardButton.No)
            if choice == QtW.QMessageBox.StandardButton.Yes:
                MaskHistoryModel.instance().clear_history()
                MaskHistoryModel.instance().delete_mask()
                MaskHistoryModel.instance().commit()
                self.ui.captionTed.setPlainText("")

                self.__updateCanvas()
        return f

    def __resetMask(self):
        @Slot()
        def f():
            MaskHistoryModel.instance().clear_history()

            self.__updateCanvas()
        return f

    def __clearMask(self):
        @Slot()
        def f():
            MaskHistoryModel.instance().delete_mask()
            MaskHistoryModel.instance().commit()

            self.__updateCanvas()
        return f

    def __undo(self):
        @Slot()
        def f():
            MaskHistoryModel.instance().undo()

            self.__updateCanvas()
        return f

    def __redo(self):
        @Slot()
        def f():
            MaskHistoryModel.instance().redo()

            self.__updateCanvas()
        return f

    def __saveMask(self):
        @Slot()
        def f():
            choice, new_mask, mask_path = self.__checkMaskChanged()
            if choice == QtW.QMessageBox.StandardButton.Yes:
                new_mask_img = Image.fromarray(new_mask, "L")
                new_mask_img.convert("RGB").save(mask_path)
                MaskHistoryModel.instance().load_mask(new_mask)

        return f

    def __saveCaption(self):
        @Slot()
        def f():
            choice, new_caption = self.__checkCaptionChanged()

            if choice == QtW.QMessageBox.StandardButton.Yes:
                DatasetModel.instance().saveCaption(self.current_image_path, new_caption)
        return f

    def __deleteCaption(self):
        @Slot()
        def f():
            if self.ui.captionTed.toPlainText().strip() != "":
                choice = self._openAlert(QCA.translate("dataset_window", "Delete Caption"),
                                        QCA.translate("dataset_window", "Do you want to delete caption?"),
                                        type="question",
                                        buttons=QtW.QMessageBox.StandardButton.Yes | QtW.QMessageBox.StandardButton.No)
                if choice == QtW.QMessageBox.StandardButton.Yes:
                    DatasetModel.instance().deleteCaption(self.current_image_path)
                    self.ui.captionTed.setPlainText("")
        return f

    def __resetCaption(self):
        @Slot()
        def f():
            _, _, caption = DatasetModel.instance().getSample(self.current_image_path)
            if caption is not None:
                self.ui.captionTed.setPlainText(caption.strip())
        return f

    def __deleteSample(self):
        @Slot()
        def f():
            if self.current_image_path is not None:
                choice = self._openAlert(QCA.translate("dataset_window", "Delete Sample"),
                                            QCA.translate("dataset_window", "Do you really want to delete the sample (image, mask and caption)? This is not reversible."),
                                            type="warning",
                                            buttons=QtW.QMessageBox.StandardButton.Yes | QtW.QMessageBox.StandardButton.No)
                if choice == QtW.QMessageBox.StandardButton.Yes:
                    DatasetModel.instance().deleteSample(self.current_image_path)
                    self.__updateDataset()()
                    self.__selectFile()()
                    self.ui.fileTreeWdg.setCurrentItem(self.leafWidgets[self.current_index])

        return f

    def __selectFile(self):
        @Slot()
        def f():
            selected_wdg = self.ui.fileTreeWdg.selectedItems()
            if len(selected_wdg) > 0:
                if self.__saveChanged():
                    self.current_image_path = selected_wdg[0].fullpath

                    idx = selected_wdg[0].idx
                    if self.current_image_path is not None:
                        if self.num_files > 0:
                            if idx is not None:
                                self.current_index = idx
                                self.ui.numFilesLbl.setText(f"{self.current_index + 1}/{self.num_files}")

                                self.image, mask, caption = DatasetModel.instance().getSample(self.current_image_path)

                                if caption is not None:
                                    self.ui.captionTed.setPlainText(caption)

                                if mask is None:
                                    mask = Image.new("L", self.image.size, 255)

                                MaskHistoryModel.instance().load_mask(np.asarray(mask))
                                self.im = self.ax.imshow(self.image)

                                self.__updateCanvas()
                else:
                    self.ui.fileTreeWdg.setCurrentItem(self.leafWidgets[self.current_index])


        return f

    def __browse(self):
        @Slot()
        def f():
            path = DatasetModel.instance().get_state("path")
            if path is not None:
                self._browse(path)
            else:
                self._openAlert(QCA.translate("dataset_window", "No Dataset Loaded"),
                                QCA.translate("dataset_window", "Please open a dataset first"))
        return f

    def __onClicked(self):
        @Slot(MouseButton, int, int)
        def f(btn, x, y):
            if btn == MouseButton.MIDDLE:
                MaskHistoryModel.instance().delete_mask() # This click is also associated with a release, which will commit the change and update the canvas.
        return f

    def __onReleased(self):
        @Slot(MouseButton, int, int)
        def f(btn, x, y):
            MaskHistoryModel.instance().commit()

            self.__updateCanvas()
        return f

    def __onWheelUp(self):
        @Slot()
        def f():
            wdg = self.canvas.toolbar.findChild(QtW.QSpinBox, "brush_sbx")
            new_val = wdg.value() + wdg.singleStep()
            wdg.setValue(new_val) # This will emit valueChanged, which is connected to self.__setBrushSize()
        return f

    def __onWheelDown(self):
        @Slot()
        def f():
            wdg = self.canvas.toolbar.findChild(QtW.QSpinBox, "brush_sbx")
            new_val = wdg.value() - wdg.singleStep()
            wdg.setValue(new_val)
        return f

    def __onMaskClicked(self):
        @Slot(MouseButton, int, int)
        def f(btn, x, y):
            if btn == MouseButton.LEFT:
                MaskHistoryModel.instance().fill(x, y, 0)
            elif btn == MouseButton.RIGHT:
                MaskHistoryModel.instance().fill(x, y, 255)
            self.__updateCanvas()
        return f

    def __onDrawMoved(self):
        @Slot(MouseButton, int, int, int, int)
        def f(btn, x0, y0, x1, y1):
            if x0 >= 0 and y0 >= 0 and x1 >= 0 and y1 >= 0:
                if btn == MouseButton.LEFT:
                    MaskHistoryModel.instance().paint_stroke(x0, y0, x1, y1, int(self.brush), 0, commit=False)  # Draw stroke 0 from x0,y0 to x1,y1
                    self.__updateCanvas(blitbb=(x0 - self.brush, x1 + self.brush, y0 - self.brush, y1 + self.brush))
                elif btn == MouseButton.RIGHT:
                    MaskHistoryModel.instance().paint_stroke(x0, y0, x1, y1, int(self.brush), 255, commit=False)
                    self.__updateCanvas(blitbb=(x0 - self.brush, x1 + self.brush, y0 - self.brush, y1 + self.brush))

        return f

    ###Utils###

    def __scan(self):
        def f(dir):
            DatasetModel.instance().set_state("path", dir)
            DatasetModel.instance().scan()
        return f

    def __saveChanged(self):
        if self.current_image_path is not None:
            choice, new_caption = self.__checkCaptionChanged(cancel=True)
            if choice != QtW.QMessageBox.StandardButton.Cancel:
                if choice == QtW.QMessageBox.StandardButton.Yes:
                    DatasetModel.instance().saveCaption(self.current_image_path, new_caption)

                choice, new_mask, mask_path = self.__checkMaskChanged(cancel=True)
                if choice != QtW.QMessageBox.StandardButton.Cancel:
                    if choice == QtW.QMessageBox.StandardButton.Yes:
                        Image.fromarray(new_mask, "L").convert("RGB").save(mask_path)

            return choice != QtW.QMessageBox.StandardButton.Cancel
        else:
            return True

    def __updateCanvas(self, blitbb=None):
        if self.im is not None:
            mask = np.clip(MaskHistoryModel.instance().get_state("current_mask")[..., np.newaxis].astype(float), 1 - self.alpha, 1)
            self.im.set_data((np.asarray(self.image) * mask).astype(np.uint8))

            if blitbb is not None:
                self.canvas.blit(Bbox.from_extents(*blitbb))

            self.canvas.draw_idle()


    def __buildTree(self, fullname, tree, idx, name=None):
        if name is None:
            name = fullname
        path = name.split("/")
        if len(path) == 1:
            tree[path[0]] = (idx, fullname)
        elif len(path) > 1:
            if path[0] not in tree:
                tree[path[0]] = {}
            self.__buildTree(fullname, tree[path[0]], idx, "/".join(path[1:]))

    def __drawTree(self, parent, tree):
        for k in sorted(tree.keys(), key=lambda x: DatasetModel.natural_sort_key(x)):
            v = tree[k]
            wdg = QtW.QTreeWidgetItem(parent, [k])
            if isinstance(v, dict):
                wdg.setIcon(0, QtG.QIcon(f"resources/icons/buttons/{self.theme}/folder-open.svg"))
                wdg.fullpath = None
                wdg.idx = None
                self.__drawTree(wdg, v)
            else:
                # TODO: We may decide to add icons to files, to show at a glance which images are associated with captions and/or masks. It requires to change DatasetModel.config.files to a triple (image file, mask file, caption file), possibly causing heavy changes to multiple methods.
                # wdg.setIcon(0, QtG.QIcon("resources/icons/buttons/{}/???.svg".format(self.theme)))
                # setTooltip to tell whether captions and masks exist
                #
                # A proposal of icons:
                # file-x-corner -> img only ("No caption or mask found")
                # file-scan -> img + mask ("Missing caption")
                # file-minus-corner -> img + caption ("Missing mask")
                # file-check-corner -> img + mask + caption ("Caption and mask available")

                wdg.fullpath = v[1]
                wdg.idx = v[0]
                self.leafWidgets[v[0]] = wdg

    def __checkMaskChanged(self, cancel=False):
        buttons = QtW.QMessageBox.StandardButton.Yes | QtW.QMessageBox.StandardButton.No
        if cancel:
            buttons |= QtW.QMessageBox.StandardButton.Cancel

        mask = MaskHistoryModel.instance().get_state("original_mask")
        new_mask = MaskHistoryModel.instance().get_state("current_mask")
        mask_path, mask_exists = DatasetModel.instance().getMaskPath(self.current_image_path)

        choice = QtW.QMessageBox.StandardButton.No
        if not mask_exists:
            choice = QtW.QMessageBox.StandardButton.Yes
        elif np.not_equal(mask, new_mask).any():
            choice = self._openAlert(QCA.translate("dataset_window", "Save Mask"),
                                    QCA.translate("dataset_window", "Mask has changed. Do you want to save it?"),
                                    type="question",
                                    buttons=buttons)
        return choice, new_mask, mask_path

    def __checkCaptionChanged(self, cancel=False):
        buttons = QtW.QMessageBox.StandardButton.Yes | QtW.QMessageBox.StandardButton.No
        if cancel:
            buttons |= QtW.QMessageBox.StandardButton.Cancel

        _, _, caption = DatasetModel.instance().getSample(self.current_image_path)
        new_caption = self.ui.captionTed.toPlainText()

        choice = QtW.QMessageBox.StandardButton.No

        if caption is None:
            choice = QtW.QMessageBox.StandardButton.Yes
        elif caption.strip() != new_caption.strip():
            choice = self._openAlert(QCA.translate("dataset_window", "Save Caption"),
                                    QCA.translate("dataset_window", "Caption has changed. Do you want to save it?"),
                                    type="question",
                                    buttons=buttons)

        return choice, new_caption
