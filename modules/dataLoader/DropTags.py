from functools import lru_cache, partial

from mgds.pipelineModules.DropTags import DropTags as MGDSDropTags


class DropTags(MGDSDropTags):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MGDS' class-level cache keys include self, retaining the entire dataset pipeline and its model.
        # Keep the same parser and cache bound, but let the cache be collected with this pipeline instance.
        load_special_tags = MGDSDropTags.get_special_tags.__wrapped__
        self.get_special_tags = lru_cache(maxsize=128)(partial(load_special_tags, self))
