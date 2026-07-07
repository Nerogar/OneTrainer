class GenerateCaptionsWindowController:
    def __init__(self, parent):
        self.parent = parent
        self.view = None

    def create_window(self, parent_window, path, parent_include_subdirectories, view_cls):
        self.view = view_cls(parent_window, self, path, parent_include_subdirectories)
        return self.view

    def create_captions(self, model_name, path, initial_caption, caption_prefix, caption_postfix, mode_str, include_subdirectories):
        self.parent.load_captioning_model(model_name)

        mode = {
            "Replace all captions": "replace",
            "Create if absent": "fill",
            "Add as new line": "add",
        }[mode_str]

        self.parent.captioning_model.caption_folder(
            sample_dir=path,
            initial_caption=initial_caption,
            caption_prefix=caption_prefix,
            caption_postfix=caption_postfix,
            mode=mode,
            progress_callback=self.view.set_progress,
            include_subdirectories=include_subdirectories,
        )
        self.parent.load_image()

    def create_captions_lmstudio(self, server_url, system_prompt, user_prompt, path, mode_str,
                                 include_subdirectories, progress_callback=None, error_callback=None,
                                 is_cancelled=None):
        self.parent.load_lmstudio_captioning_model(server_url, system_prompt, user_prompt)

        mode = {
            "Replace all captions": "replace",
            "Create if absent": "fill",
            "Add as new line": "add",
        }[mode_str]

        self.parent.captioning_model.caption_folder(
            sample_dir=path,
            mode=mode,
            progress_callback=progress_callback,
            error_callback=error_callback,
            include_subdirectories=include_subdirectories,
            is_cancelled=is_cancelled,
        )
