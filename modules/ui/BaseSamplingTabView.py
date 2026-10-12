from abc import ABC, abstractmethod

from modules.ui.BaseConfigListView import BaseConfigListView


class BaseSamplingTabView(BaseConfigListView):
    pass


class BaseSampleWidgetView(ABC):
    def __init__(self, components):
        self.components = components

    def build_content(self, frame, element, ui_state, i, open_command, remove_command, clone_command, save_command):
        self.element = element
        self.i = i
        self.save_command = save_command

        # close button
        self.components.colored_icon_button(frame, 0, 0, "X", "#C00000", lambda: remove_command(self.i))

        # clone button
        self.components.colored_icon_button(frame, 0, 1, "+", "#00C000", lambda: clone_command(self.i), padx=5)

        # enabled
        self.enabled_switch = self.components.switch(frame, 0, 2, ui_state, "enabled", self._switch_enabled, width=40)

        # width
        self.components.label(frame, 0, 3, "width:")
        self.width_entry = self.components.entry(frame, 0, 4, ui_state, "width", width=50)

        # height
        self.components.label(frame, 0, 5, "height:")
        self.height_entry = self.components.entry(frame, 0, 6, ui_state, "height", width=50)

        # seed
        self.components.label(frame, 0, 7, "seed:")
        self.seed_entry = self.components.entry(frame, 0, 8, ui_state, "seed", width=80)

        # prompt
        self.components.label(frame, 0, 9, "prompt:")
        self.prompt_entry = self.components.entry(frame, 0, 10, ui_state, "prompt")

        # button
        self.button = self.components.icon_button(frame, 0, 11, "...", lambda: open_command(self.i, ui_state))

        self._bind_save(save_command)
        self._set_enabled()

    @abstractmethod
    def _bind_save(self, save_command): pass

    # BaseConfigListView calls configure_element() on all widget types generically;
    # sampling widgets have no post-window logic, so this is an intentional no-op.
    def configure_element(self): pass  # noqa: B027

    def _switch_enabled(self):
        self.save_command()
        self._set_enabled()

    def _set_enabled(self):
        enabled = self.element.enabled
        self.width_entry.configure(state="normal" if enabled else "disabled")
        self.height_entry.configure(state="normal" if enabled else "disabled")
        self.prompt_entry.configure(state="normal" if enabled else "disabled")
        self.seed_entry.configure(state="normal" if enabled else "disabled")
        self.button.configure(state="normal" if enabled else "disabled")
