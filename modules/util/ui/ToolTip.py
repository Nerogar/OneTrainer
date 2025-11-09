import contextlib

import customtkinter as ctk


class ToolTip:
    MAX_TOOLTIP_LENGTH = 350

    def __init__(self, widget, text='widget info', x_position=20, wide=False,
                 hover_only=True, track_movement=False):
        self.widget = widget
        self.text = text
        self.x_position = x_position
        self.wraplength = 350 if wide else 180
        self.hover_only = hover_only
        self.track_movement = track_movement

        self.waittime = 500
        self.id = None
        self.tw = None
        self._after_id = None

        if hover_only:
            self.widget.bind("<Enter>", self.enter)
            self.widget.bind("<Leave>", self.leave)
            self.widget.bind("<ButtonPress>", self.leave)

        if track_movement:
            self._root = widget.winfo_toplevel()
            self._root.bind("<Configure>", self._on_move, add="+")
            self._root.bind("<FocusOut>", self.hide, add="+")
            self._root.bind("<Unmap>", self.hide, add="+")

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hide()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def _cancel_scheduled_hide(self):
        if self._after_id:
            with contextlib.suppress(Exception):
                self.widget.after_cancel(self._after_id)
            self._after_id = None

    def show(self, text=None, color=None, duration_ms=None, fg_color="#2b2b2b"):
        if text:
            self.text = text[:self.MAX_TOOLTIP_LENGTH] + ("..." if len(text) > self.MAX_TOOLTIP_LENGTH else "")

        self._cancel_scheduled_hide()
        self.hide()

        self.tw = ctk.CTkToplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        if self.track_movement:
            self.tw.attributes("-topmost", True)

        frame = ctk.CTkFrame(self.tw, fg_color=fg_color, corner_radius=6)
        frame.pack()

        label = ctk.CTkLabel(
            frame,
            text=self.text,
            justify='left',
            wraplength=self.wraplength,
            text_color=color or "white",
            fg_color="transparent",
            padx=8,
            pady=6 if self.track_movement else 8
        )
        label.pack()

        self._position()

        if duration_ms:
            self._after_id = self.widget.after(duration_ms, self.hide)

    def showtip(self, event=None):
        self.show()

    def _position(self):
        if not self.tw or not self.tw.winfo_exists():
            return

        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + self.x_position

        if self.track_movement:
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4

            self.tw.update_idletasks()
            tip_width = self.tw.winfo_reqwidth()
            tip_height = self.tw.winfo_reqheight()
            screen_width = self.widget.winfo_screenwidth()
            screen_height = self.widget.winfo_screenheight()

            x = max(10, min(self.widget.winfo_rootx(), screen_width - tip_width - 10))

            if y + tip_height > screen_height - 10:
                y = self.widget.winfo_rooty() - tip_height - 4

            y = max(10, y)

        self.tw.wm_geometry(f"+{x}+{y}")

    def _on_move(self, _evt=None):
        if self.tw and self.tw.winfo_exists():
            self._position()

    def hide(self, _evt=None):
        self._cancel_scheduled_hide()

        if self.tw and self.tw.winfo_exists():
            self.tw.destroy()
        self.tw = None

    def show_error(self, message: str, duration_ms: int | None = 7000):
        self.show(message, color="#ff6b6b", duration_ms=duration_ms)

    def show_warning(self, message: str, duration_ms: int | None = 7000):
        self.show(message, color="#ff9500", duration_ms=duration_ms)

    def show_info(self, message: str, duration_ms: int | None = 5000):
        self.show(message, color="#5bc0de", duration_ms=duration_ms)

    def is_visible(self) -> bool:
        return self.tw is not None and self.tw.winfo_exists()

    def destroy(self):
        self.hide()
        if self.track_movement:
            with contextlib.suppress(Exception):
                self._root.unbind("<Configure>", self._on_move)
                self._root.unbind("<FocusOut>", self.hide)
                self._root.unbind("<Unmap>", self.hide)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.destroy()
