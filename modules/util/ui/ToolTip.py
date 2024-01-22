import customtkinter as ctk


class ToolTip(object):
    """
    create a tooltip for a given widget
    """

    def __init__(self, widget, text='widget info', x_position=20, wide=False):
        self.widget = widget
        self.text = text
        self.x_position = x_position

        self.waittime = 500  # miliseconds
        self.wraplength = 180 if not wide else 350 # pixels
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + self.x_position
        # creates a toplevel window
        self.tw = ctk.CTkToplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = ctk.CTkLabel(self.tw, text=self.text, justify='left', wraplength=self.wraplength)
        label.pack(padx=8, pady=8)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()
