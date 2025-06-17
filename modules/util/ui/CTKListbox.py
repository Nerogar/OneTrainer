import ast

import customtkinter


class CTkListbox(customtkinter.CTkFrame):
    def _resolve_color(self, color):
        # If color is a tuple/list, pick the first entry
        if isinstance(color, (tuple | list)):
            return color[0]
        return color

    def __init__(
        self,
        master: any,
        height: int = 100,
        width: int = 150,
        highlight_color: str = "default",
        fg_color: str = "transparent",
        bg_color: str = None,
        text_color: str = "default",
        hover_color: str = "default",
        button_color: str = "default",
        border_width: int = 3,
        font: tuple = None,
        multiple_selection: bool = False,
        listvariable=None,
        hover: bool = True,
        command=None,
        wraplength=0,
        justify="left",
        **kwargs,
    ):
        super().__init__(
            master,
            width=width,
            height=height,
            fg_color=fg_color,
            border_width=border_width,
            **kwargs,
        )

        self._canvas = customtkinter.CTkCanvas(self, width=width, height=height, highlightthickness=0, bg=fg_color)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._scrollbar = customtkinter.CTkScrollbar(self, orientation="vertical", command=self._on_scroll)
        self._scrollbar.grid(row=0, column=1, sticky="ns")
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)

        if bg_color:
            self.configure(bg_color=bg_color)

        self.select_color = self._resolve_color(
            customtkinter.ThemeManager.theme["CTkButton"]["fg_color"]
            if highlight_color == "default"
            else highlight_color
        )
        self.text_color = self._resolve_color(
            customtkinter.ThemeManager.theme["CTkLabel"]["text_color"]
            if text_color == "default"
            else text_color
        )
        self.hover_color = self._resolve_color(
            customtkinter.ThemeManager.theme["CTkButton"]["hover_color"]
            if hover_color == "default"
            else hover_color
        )

        if not font:
            self.font = customtkinter.CTkFont(customtkinter.ThemeManager.theme["CTkFont"]["family"], 13)
        else:
            if isinstance(font, customtkinter.CTkFont):
                self.font = font
            elif isinstance(font, tuple):
                self.font = customtkinter.CTkFont(*font)
            else:
                self.font = customtkinter.CTkFont(customtkinter.ThemeManager.theme["CTkFont"]["family"], 13)

        self.button_fg_color = self._resolve_color(
            self.cget("bg_color") if button_color == "default" or button_color == "transparent" else button_color
        )

        self.justify = {"left": "w", "right": "e"}.get(justify, "c")
        self.command = command
        self.multiple = multiple_selection
        self.hover = hover
        self.wraplength = wraplength

        self._item_height = 36  # Adjust for your visuals
        self._items = []  # List of dicts: {"text": ..., "rect": ..., "text_id": ...}
        self._selected_indices = set()
        self._hover_index = None

        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<Motion>", self._on_motion)
        self._canvas.bind("<Leave>", self._on_leave)
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        if listvariable:
            self.listvariable = listvariable
            self.listvariable.trace_add("write", lambda a, b, c: self.update_listvar())
            self.update_listvar()

    def insert(self, index, option, update=True, **args):
        if str(index).lower() == "end":
            index = len(self._items)
        self._items.insert(index, {"text": option})
        self._redraw_items()

    def insert_many(self, options):
        self._items.clear()
        for option in options:
            self._items.append({"text": option})
        self._selected_indices.clear()
        self._redraw_items()

    def update_listvar(self):
        values = list(ast.literal_eval(self.listvariable.get()))
        self.insert_many(values)

    def delete(self, index, last=None):
        if str(index).lower() == "all":
            self._items.clear()
            self._selected_indices.clear()
            self._redraw_items()
            return
        if str(index).lower() == "end":
            index = len(self._items) - 1
        if last is not None:
            if str(last).lower() == "end":
                last = len(self._items) - 1
            del self._items[index:last+1]
        else:
            del self._items[index]
        self._selected_indices = {i for i in self._selected_indices if i < len(self._items)}
        self._redraw_items()

    def select(self, index):
        if str(index).lower() == "all":
            if self.multiple:
                self._selected_indices = set(range(len(self._items)))
            self._redraw_items()
            return
        if not self.multiple:
            self._selected_indices = {index}
        else:
            if index in self._selected_indices:
                self._selected_indices.remove(index)
            else:
                self._selected_indices.add(index)
        self._redraw_items()
        if self.command:
            self.command(self.get())
        self.event_generate("<<ListboxSelect>>")

    def deselect(self, index):
        self._selected_indices.discard(index)
        self._redraw_items()

    def deactivate(self, index):
        self.deselect(index)

    def activate(self, index):
        self.select(index)

    def curselection(self):
        return tuple(sorted(self._selected_indices))

    def get(self, index=None):
        if index is not None:
            if str(index).lower() == "all":
                return [item["text"] for item in self._items]
            else:
                return self._items[index]["text"]
        else:
            if self.multiple:
                return [self._items[i]["text"] for i in self._selected_indices] if self._selected_indices else None
            else:
                if self._selected_indices:
                    return self._items[next(iter(self._selected_indices))]["text"]
                return None

    def size(self):
        return len(self._items)

    def see(self, index):
        y = index * self._item_height
        self._canvas.yview_moveto(y / max(1, self._canvas.bbox("all")[3]))

    def configure(self, **kwargs):
        if "hover_color" in kwargs:
            self.hover_color = kwargs.pop("hover_color")
        if "button_color" in kwargs:
            self.button_fg_color = kwargs.pop("button_color")
        if "highlight_color" in kwargs:
            self.select_color = kwargs.pop("highlight_color")
        if "text_color" in kwargs:
            self.text_color = kwargs.pop("text_color")
        if "font" in kwargs:
            self.font = kwargs.pop("font")
        if "command" in kwargs:
            self.command = kwargs.pop("command")
        if "hover" in kwargs:
            self.hover = kwargs.pop("hover")
        if "justify" in kwargs:
            self.justify = {"left": "w", "right": "e"}.get(kwargs.pop("justify"), "c")
        if "height" in kwargs:
            self._canvas.configure(height=kwargs["height"])
        if "multiple_selection" in kwargs:
            self.multiple = kwargs.pop("multiple_selection")
        super().configure(**kwargs)
        self._redraw_items()

    def cget(self, param):
        if param == "hover_color":
            return self.hover_color
        if param == "button_color":
            return self.button_fg_color
        if param == "highlight_color":
            return self.select_color
        if param == "text_color":
            return self.text_color
        if param == "font":
            return self.font
        if param == "hover":
            return self.hover
        if param == "justify":
            return self.justify
        return super().cget(param)

    def move_up(self, index):
        if index > 0:
            self._items[index - 1], self._items[index] = self._items[index], self._items[index - 1]
            sel = set()
            for i in self._selected_indices:
                if i == index:
                    sel.add(i - 1)
                elif i == index - 1:
                    sel.add(i + 1)
                else:
                    sel.add(i)
            self._selected_indices = sel
            self._redraw_items()

    def move_down(self, index):
        if index < len(self._items) - 1:
            self._items[index + 1], self._items[index] = self._items[index], self._items[index + 1]
            sel = set()
            for i in self._selected_indices:
                if i == index:
                    sel.add(i + 1)
                elif i == index + 1:
                    sel.add(i - 1)
                else:
                    sel.add(i)
            self._selected_indices = sel
            self._redraw_items()

    def _redraw_item(self, idx):
        if not (0 <= idx < len(self._items)):
            return
        width = self._canvas.winfo_width() or int(self._canvas["width"])
        y0_item = idx * self._item_height
        y1_item = y0_item + self._item_height
        fill = self.select_color if idx in self._selected_indices else self.button_fg_color
        if self.hover and idx == self._hover_index and idx not in self._selected_indices:
            fill = self.hover_color
        item = self._items[idx]
        if "rect" in item:
            self._canvas.delete(item["rect"])
        if "text_id" in item:
            self._canvas.delete(item["text_id"])
        # Adjust padding and wrapping here
        padding = 2
        anchor = "w"
        x = padding
        item["rect"] = self._canvas.create_rectangle(0, y0_item, width, y1_item, fill=fill, outline="")
        item["text_id"] = self._canvas.create_text(
            x,
            y0_item + self._item_height // 2,
            anchor=anchor,
            text=item["text"],
            fill=self.text_color,
            font=self.font,
            width=width - 2 * padding,  # wrap at canvas width minus padding
        )

    def _redraw_items(self):
        self._canvas.delete("all")
        width = self._canvas.winfo_width() or int(self._canvas["width"])
        height = self._canvas.winfo_height() or int(self._canvas["height"])
        y0 = int(self._canvas.canvasy(0))
        y1 = y0 + height
        first_idx = max(0, y0 // self._item_height)
        last_idx = min(len(self._items), (y1 // self._item_height) + 1)
        padding = 2
        anchor = "w"
        x = padding
        for i in range(first_idx, last_idx):
            item = self._items[i]
            y0_item = i * self._item_height
            y1_item = y0_item + self._item_height
            fill = self.select_color if i in self._selected_indices else self.button_fg_color
            if self.hover and i == self._hover_index and i not in self._selected_indices:
                fill = self.hover_color
            rect = self._canvas.create_rectangle(0, y0_item, width, y1_item, fill=fill, outline="")
            text_id = self._canvas.create_text(
                x,
                y0_item + self._item_height // 2,
                anchor=anchor,
                text=item["text"],
                fill=self.text_color,
                font=self.font,
                width=width - 2 * padding,  # wrap at canvas width minus padding
            )
            item["rect"] = rect
            item["text_id"] = text_id
        self._canvas.config(scrollregion=(0, 0, width, len(self._items) * self._item_height))

    def _redraw_visible_items(self):
        width = self._canvas.winfo_width() or int(self._canvas["width"])
        height = self._canvas.winfo_height() or int(self._canvas["height"])
        y0 = int(self._canvas.canvasy(0))
        y1 = y0 + height
        first_idx = max(0, y0 // self._item_height)
        last_idx = min(len(self._items), (y1 // self._item_height) + 1)
        # Remove items not in visible range
        for i, item in enumerate(self._items):
            if "rect" in item and not (first_idx <= i < last_idx):
                self._canvas.delete(item["rect"])
                self._canvas.delete(item["text_id"])
                item.pop("rect", None)
                item.pop("text_id", None)
        # Draw only visible items
        for i in range(first_idx, last_idx):
            self._redraw_item(i)
        self._canvas.config(scrollregion=(0, 0, width, len(self._items) * self._item_height))

    def _on_scroll(self, *args):
        self._canvas.yview(*args)
        self._redraw_visible_items()

    def _on_motion(self, event):
        y = self._canvas.canvasy(event.y)
        idx = int(y // self._item_height)
        if idx != self._hover_index:
            old_hover = self._hover_index
            self._hover_index = idx if 0 <= idx < len(self._items) else None
            if old_hover is not None:
                self._redraw_item(old_hover)
            if self._hover_index is not None:
                self._redraw_item(self._hover_index)

    def _on_click(self, event):
        y = self._canvas.canvasy(event.y)
        idx = int(y // self._item_height)
        if 0 <= idx < len(self._items):
            old_selection = set(self._selected_indices)
            self.select(idx)
            # Only redraw changed items
            changed = old_selection ^ self._selected_indices
            for i in changed:
                self._redraw_item(i)
            self._redraw_item(idx)

    def _on_leave(self, event):
        if self._hover_index is not None:
            old_hover = self._hover_index
            self._hover_index = None
            self._redraw_item(old_hover)

    def _on_mousewheel(self, event):
        if event.widget == self._canvas:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            self._redraw_items()
