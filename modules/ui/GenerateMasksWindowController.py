class GenerateMasksWindowController:
    def __init__(self, parent):
        self.parent = parent
        self.view = None

    def create_window(self, parent_window, path, parent_include_subdirectories, view_cls):
        self.view = view_cls(parent_window, self, path, parent_include_subdirectories)
        return self.view

    def create_masks(self, model_name, path, prompt, mode_str, alpha_str, threshold_str, smooth_str, expand_str, include_subdirectories):
        self.parent.load_masking_model(model_name)

        mode = {
            "Replace all masks": "replace",
            "Create if absent": "fill",
            "Add to existing": "add",
            "Subtract from existing": "subtract",
            "Blend with existing": "blend",
        }[mode_str]

        self.parent.masking_model.mask_folder(
            sample_dir=path,
            prompts=[prompt],
            mode=mode,
            alpha=float(alpha_str),
            threshold=float(threshold_str),
            smooth_pixels=int(smooth_str),
            expand_pixels=int(expand_str),
            progress_callback=self.view.set_progress,
            include_subdirectories=include_subdirectories,
        )
        self.parent.load_image()
