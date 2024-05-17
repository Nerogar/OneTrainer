from pathlib import Path

import customtkinter as ctk

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConfigPart import ConfigPart
from modules.util.replacement_util import parse_directory_for_folders, process_parsed_directories, replace_text_in_trainconfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class AutomationTab:

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super(AutomationTab, self).__init__()
        self.init_load = True
        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state

        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.scroll_frame = None

        self.__setup_ui()

    def refresh_ui(self):
        if self.init_load_finished:
            if self.scroll_frame:
                self.scroll_frame.destroy()
            self.__setup_ui()
        
    def __setup_ui(self):
        self.scroll_frame = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")

        self.scroll_frame.grid_columnconfigure(0, weight=0)
        self.scroll_frame.grid_columnconfigure(1, weight=10)
        self.scroll_frame.grid_columnconfigure(2, minsize=50)
        self.scroll_frame.grid_columnconfigure(3, weight=0)
        self.scroll_frame.grid_columnconfigure(4, weight=1)
        self.init_load = False
        
        row = 0
        # row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            automation_enabled=True,
            automation_replacement_keyword="",
            automation_replacement_text=""
        )
        
        self.init_load_finished = True;

    def __create_base_components(
            self,
            row: int,
            automation_enabled: bool = False,
            automation_replacement_keyword: str = "",
            automation_replacement_text: str = ""
    ) -> int:
        components.label(self.scroll_frame, row, 0, "Enabled Directory Automation",
                        tooltip="Enable replacement automation through {{replacement text}}")
        components.switch(self.scroll_frame, row, 1, self.ui_state, "automation_directory_enabled")

        row += 1
        components.label(self.scroll_frame, row, 0, "Queued Work Directory",
                            tooltip="The directory where queued work is organized into folders that can have keywords in the folder name.")
        components.dir_entry(self.scroll_frame, row, 1, self.ui_state, "automation_queued_dir")
            
        row += 1
        components.label(self.scroll_frame, row, 0, "Test Dir Parsing",
                        tooltip="Test the parsing of the directory and string replacement")
        components.button(self.scroll_frame, row, 1, "Test Dir", self.test_dir_replacement)
        
    def test_dir_replacement(self):
        directories = parse_directory_for_folders(self.train_config.automation_queued_dir, self.train_config)
        results = process_parsed_directories(directories, self.train_config, True)
        for r in results:
            print(r.to_dict())
            

       