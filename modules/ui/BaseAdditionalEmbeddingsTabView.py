
from modules.ui.BaseConfigListView import BaseConfigListView
from modules.util import path_util


class BaseAdditionalEmbeddingsTabView(BaseConfigListView):

    def refresh_ui(self):
        if self.element_list is not None:
            self._destroy_frame(self.element_list)
            self.element_list = None
        self.widgets_initialized = False
        self._create_element_list()

    def open_element_window(self, i, ui_state):
        pass


class BaseEmbeddingWidgetView:

    def __init__(self, components):
        self.components = components

    def build_content(self, top_frame, bottom_frame, ui_state, i, save_command, remove_command, clone_command, controller):
        self.ui_state = ui_state
        self.i = i
        self.save_command = save_command

        # close button
        self.components.colored_icon_button(top_frame, 0, 0, "X", "#C00000", lambda: remove_command(self.i))

        # clone button
        self.components.colored_icon_button(top_frame, 0, 1, "+", "#00C000", lambda: clone_command(self.i, controller.randomize_uuid), padx=5)

        # embedding model names
        self.components.label(top_frame, 0, 2, "base embedding:",
                              tooltip="训练的基础嵌入，留空创建新嵌入")
        self.components.path_entry(
            top_frame, 0, 3, self.ui_state, "model_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # placeholder
        self.components.label(top_frame, 0, 4, "placeholder:",
                              tooltip="在提示词中使用嵌入的占位符")
        self.components.entry(top_frame, 0, 5, self.ui_state, "placeholder")

        # token count
        self.components.label(top_frame, 0, 6, "token count:",
                              tooltip="新嵌入的Token数，留空自动检测")
        self.components.entry(top_frame, 0, 7, self.ui_state, "token_count", width=40)

        # trainable
        self.components.label(bottom_frame, 0, 0, "train:")
        self.components.switch(bottom_frame, 0, 1, self.ui_state, "train", command=save_command, width=40)

        # output embedding
        self.components.label(bottom_frame, 0, 2, "output embedding:",
                              tooltip="在文本编码器输出处计算嵌入，可改善大文本编码器效果并降低显存")
        self.components.switch(bottom_frame, 0, 3, self.ui_state, "is_output_embedding", width=40)

        # stop training after
        self.components.label(bottom_frame, 0, 4, "stop training after:",
                              tooltip="何时停止训练嵌入")
        self.components.time_entry(bottom_frame, 0, 5, self.ui_state, "stop_training_after", "stop_training_after_unit")

        # initial embedding text
        self.components.label(bottom_frame, 0, 6, "initial embedding text:",
                              tooltip="创建新嵌入时的初始文本")
        self.components.entry(bottom_frame, 0, 7, self.ui_state, "initial_embedding_text")

    def configure_element(self):
        pass
