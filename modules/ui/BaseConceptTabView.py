import os
import pathlib

from modules.ui.BaseConfigListView import BaseConfigListView
from modules.ui.ConceptWindowController import ConceptWindowController
from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.image_util import load_image

from PIL import Image


class BaseConceptTabView(BaseConfigListView):

    _FILTER_TYPES = ["ALL", "STANDARD", "VALIDATION", "PRIOR_PREDICTION"]

    def _element_matches_filters(self, element):
        if not self.filters.get("show_disabled", True):
            if hasattr(element, 'enabled') and not element.enabled:
                return False

        search = self.filters.get("search", "").lower()
        if search:
            if not hasattr(element, '_search_cache'):
                cache = []
                try:
                    if getattr(element, 'name', None):
                        cache.append(element.name.lower())
                    p = getattr(element, 'path', None)
                    if p:
                        try:
                            cache.append(os.path.basename(p).lower())
                            cache.append(p.lower())
                        except (TypeError, AttributeError):
                            pass
                except (AttributeError, TypeError):
                    pass
                element._search_cache = cache
            if not any(search in text for text in getattr(element, '_search_cache', [])):
                return False

        type_filter = self.filters.get("type", "ALL")
        if type_filter != "ALL":
            if hasattr(element, 'type') and element.type:
                try:
                    return ConceptType(element.type).value == type_filter
                except (ValueError, AttributeError):
                    return False
            return False

        return True



class BaseConceptWidgetView:

    def __init__(self, components, concept: ConceptConfig):
        self.components = components
        self.concept = concept

    def _get_display_name(self):
        if self.concept.name:
            return self.concept.name
        elif self.concept.path:
            return os.path.basename(self.concept.path)
        else:
            return ""

    def _get_preview_image(self):
        preview_path = "resources/icons/icon.png"
        glob_pattern = "**/*.*" if getattr(self.concept, 'include_subdirectories', False) else "*.*"

        concept_path = ConceptWindowController.get_concept_path(getattr(self.concept, 'path', None))
        if concept_path:
            for path in pathlib.Path(concept_path).glob(glob_pattern):
                if any(part.startswith('.') for part in path.relative_to(concept_path).parent.parts):
                    continue
                extension = os.path.splitext(path)[1]
                if (path.is_file()
                        and path_util.is_supported_image_extension(extension)
                        and not path.name.endswith("-masklabel.png")
                        and not path.name.endswith("-condlabel.png")):
                    preview_path = path_util.canonical_join(concept_path, path)
                    break
        try:
            image = load_image(preview_path, convert_mode="RGBA")
        except OSError:
            image = Image.new("RGBA", (150, 150), (200, 200, 200, 255))
        size = min(image.width, image.height)
        image = image.crop((
            (image.width - size) // 2,
            (image.height - size) // 2,
            (image.width - size) // 2 + size,
            (image.height - size) // 2 + size,
        ))
        return image.resize((150, 150), Image.Resampling.BILINEAR)

    def _clear_search_cache(self):
        try:
            if hasattr(self.concept, '_search_cache'):
                delattr(self.concept, '_search_cache')
        except AttributeError:
            pass
