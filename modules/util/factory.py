import importlib
import pkgutil

__registry = {}

def get(base_cls, *args, **kwargs):
    entries = __registry.get(base_cls)
    if entries is None:
        return None
    for entry in entries:
        if entry[0] == args and entry[1] == kwargs:
            return entry[2]
    return None

def _do_register(base_cls, cls, *args, **kwargs):
    if get(base_cls, *args, **kwargs) is not None:
        raise RuntimeError(f"{cls} already registered as an implementation of {base_cls} with the same criteria {args} {kwargs}")

    if base_cls not in __registry:
        __registry[base_cls] = []
    __registry[base_cls].append((args, kwargs, cls))

def register(base_cls, cls_or_first_key, *args, **kwargs):
    if isinstance(cls_or_first_key, type):
        # direct call: register(Base, Cls, key1, key2)
        _do_register(base_cls, cls_or_first_key, *args, **kwargs)
        return cls_or_first_key
    else:
        # decorator: @register(Base, key1, key2)
        def decorator(cls):
            _do_register(base_cls, cls, cls_or_first_key, *args, **kwargs)
            return cls
        return decorator

def import_dir(path: str, parent: str):
    for _finder, name, _ispkg in pkgutil.walk_packages([path], parent+"."):
        importlib.import_module(name)
