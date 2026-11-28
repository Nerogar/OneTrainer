from functools import lru_cache

from mgds.pipelineModules.DropTags import DropTags as MGDSDropTags


class DropTags(MGDSDropTags):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep a per-instance cache so cached tag lists are released (and reloaded on subsequent runs)
        # with this pipeline instance rather than living in MGDS's process-global lru_cache.
        self.get_special_tags = lru_cache(maxsize=128)(MGDSDropTags.get_special_tags.__wrapped__)
