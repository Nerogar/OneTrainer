import re
from re import Pattern


class ModuleFilter:
    """
    ModuleFilter allows filtering (LoRA/LoHA/DoRA) module names using either substring matching or regular expressions.

    Args:
        pattern (str): The filter pattern.
        use_regex (bool): If True, interpret pattern as a regex; else, use substring matching.
    """
    __slots__ = ('_pattern', '_use_regex', '_compiled', '_used')

    _pattern: str
    _use_regex: bool
    _compiled: Pattern[str] | None
    _used: bool

    def __init__(self, pattern: str, use_regex: bool = False):
        assert pattern.isprintable(), f'Custom layer filter contains non-printable characters: {repr(pattern)}'
        self._pattern = pattern.strip()
        # empty patterns are allowed and will match all layers, resulting in full training
        self._use_regex = use_regex
        self._used = False
        self._compiled = None
        if self._use_regex and self._pattern:
            try:
                self._compiled = re.compile(self._pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {self._pattern!r}: {e}") from e

    def __repr__(self) -> str:
        return repr(self._pattern)

    def matches(self, module_name: str) -> bool:
        """
        Checks if the given module name matches the filter pattern.

        Args:
            module_name (str): The name of the module to check.

        Returns:
            bool: True if the module name matches, False otherwise.
        """
        is_match = self._compiled.search(module_name) is not None if self._compiled else self._pattern in module_name

        if is_match:
            self._used = True
        return is_match

    def was_used(self) -> bool:
        return self._used


def tests():
    block_names = [
        'down_blocks.1.resnets.0.conv2',
        'down_blocks',
        'mid_block',
        'up_blocks.0.attentions.1.transformer_blocks.5',
    ]
    for name in block_names:
        f = ModuleFilter(name)
        assert not f.was_used()
        _ = f.matches('up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_v') # just to hit more of the code
        assert f.matches(name)
        assert f.was_used()

    asterisk = ModuleFilter('.*', use_regex=True)
    assert not asterisk.was_used()
    for name in block_names:
        assert asterisk.matches(name)
    assert asterisk.was_used()

    f = ModuleFilter('attentions')
    assert f.matches('lora.unet.up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_v')
    assert f.was_used()
    assert f.matches('lora.unet.up_blocks.0.attentions.2.transformer_blocks.4.foo.bar')

    f = ModuleFilter('attn')
    assert not f.was_used()
    assert f.matches('lora.unet.up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_v')
    assert f.was_used()
    assert f.matches('lora.unet.down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_v')
    assert not f.matches('lora.unet.up_blocks.0.attentions.2.transformer_blocks.4.foo.bar')

    f = ModuleFilter('down_blocks.1.resnets.0.conv2')
    assert f.matches('lora.unet.down_blocks.1.resnets.0.conv2')
    assert not f.matches('lora.unet.up_blocks.1.resnets.0.conv2')
    assert f.was_used()

    f = ModuleFilter('up_blocks.0.attentions.2.transformer_blocks.[56].attn[12].to_v', use_regex=True)
    assert f.matches('lora.unet.up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_v')
    assert f.matches('lora.unet.up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_v')
    assert not f.matches('lora.unet.up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_k')
    assert not f.matches('lora.unet.up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_v')
    assert f.was_used()

    f = ModuleFilter('down_blocks.2.attentions.0.transformer_blocks.[468].attn2.to_v', use_regex=True)
    assert not f.matches('down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_v')
    assert not f.was_used()
    assert f.matches('down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_v')
    assert f.was_used()

    try:
        ModuleFilter('invalid_regex_(foo|bar', use_regex=True)
        raise AssertionError('The line above should have raised an error')
    except ValueError:
        pass

    f = ModuleFilter('up_bl..ks', use_regex=True)
    assert f.matches('up_blocks.0.attentions.1')

    f = ModuleFilter('down_blocks.2.attentions.[01].transformer_blocks.[0-9].ff.net.0.proj', use_regex=True)
    assert f.matches('lora.unet.down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj')
    assert f.matches('lora.unet.down_blocks.2.attentions.1.transformer_blocks.6.ff.net.0.proj')

    empty = ModuleFilter('')
    for name in block_names:
        assert empty.matches(name)

    my_filters = '''
    up_blocks.0.(attentions|resnets|upsamplers).[02], up_blocks.1.*(conv|time.embed|attn2.to.v), up_blocks.2.resnets.[012].(conv|time)
    '''
    filters = [ModuleFilter(pattern, use_regex=True) for pattern in my_filters.strip().split(",")]
    assert filters[0].matches('up_blocks.0.attentions.0')
    assert filters[0].matches('up_blocks.0.resnets.2')
    assert not filters[0].matches('up_blocks.1.attentions.2')
    assert filters[1].matches('up_blocks.1.foo.bar.attn2.to.v')
    assert filters[2].matches('up_blocks.2.resnets.1.conv')


def main():
    tests()
    print('All tests passed OK')


if __name__ == '__main__':
    main()
