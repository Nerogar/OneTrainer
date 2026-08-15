class BaseSampleWindowView:
    def __init__(self, components):
        self.components = components

        # every image this window has shown, in arrival order, so the user can
        # scroll back through a batch (or the whole session) with the nav arrows
        self._gallery = []
        self._gallery_index = -1

    def gallery_add(self, image):
        # a freshly produced image always becomes the shown one
        self._gallery.append(image)
        self._gallery_index = len(self._gallery) - 1

    def gallery_step(self, delta):
        self._gallery_index = max(0, min(len(self._gallery) - 1, self._gallery_index + delta))

    @property
    def gallery_current(self):
        return self._gallery[self._gallery_index] if self._gallery else None

    @property
    def gallery_index(self):
        return self._gallery_index

    @property
    def gallery_count(self):
        return len(self._gallery)
