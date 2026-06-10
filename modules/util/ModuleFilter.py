import re
from re import Pattern

from modules.util.config.TrainConfig import TrainConfig


class ModuleFilter:
    """
    ModuleFilter allows filtering (LoRA/LoHA/DoRA) module names using either substring matching or regular expressions.

    Args:
        pattern (str): The filter pattern.
        use_regex (bool): If True, interpret pattern as a regex; else, use substring matching.
    """

    _pattern: str
    _compiled_regex: bool
    _compiled: Pattern[str] | None
    _used: bool


    def __init__(self, pattern: str, use_regex: bool = False):
        self._pattern = pattern.strip()
        if not pattern.isprintable():
            raise ValueError(f"Custom layer filter contains non-printable characters: {repr(pattern)}")

        # empty patterns are allowed and will match all layers, resulting in full training
        self._compiled_regex = False
        self._used = False
        self._compiled = None
        if use_regex and self._pattern:
            try:
                self._compiled = re.compile(self._pattern)
                self._compiled_regex = True
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {self._pattern!r}: {e}") from e

    @staticmethod
    def create(config: TrainConfig):
        return [
            ModuleFilter(pattern, use_regex=config.layer_filter_regex)
            for pattern in config.layer_filter.split(",")
        ]


    def matches(self, module_name: str) -> bool:
        """
        Checks if the given module name matches the filter pattern.

        Args:
            module_name (str): The name of the module to check.

        Returns:
            bool: True if the module name matches, False otherwise.
        """
        if not self._pattern:
            is_match = True
        else:
            is_match = self._compiled.search(module_name) is not None if self._compiled else self._pattern in module_name

        if is_match:
            self._used = True
        return is_match

    def was_used(self) -> bool:
        return self._used


def test_simple_substring_list():
    """
    Tests filtering using a comma-delimited list of simple substrings to make sure non regex works as expected
    """
    print("Running test_simple_substring_list...")

    module_names = [
        'lora.unet.down_blocks.0.attentions.0.attn1.to_v',  # Should match 'attentions'
        'lora.unet.down_blocks.1.resnets.0.conv1',          # Should match 'conv1'
        'lora.unet.down_blocks.1.resnets.0.conv2',          # Should NOT match
        'lora.unet.mid_block.resnets.0.time_emb_proj',      # Should match 'time_emb'
        'lora.unet.up_blocks.2.resnets.1.conv_shortcut',    # Should NOT match
    ]

    filter_string = "attentions, conv1, time_emb"

    filters = [ModuleFilter(p, use_regex=False) for p in filter_string.split(',')]

    # Should match
    assert any(f.matches(module_names[0]) for f in filters), f"'{module_names[0]}' should match"
    assert any(f.matches(module_names[1]) for f in filters), f"'{module_names[1]}' should match"
    assert any(f.matches(module_names[3]) for f in filters), f"'{module_names[3]}' should match"

    # Shouldnt match
    assert not any(f.matches(module_names[2]) for f in filters), f"'{module_names[2]}' should NOT match"
    assert not any(f.matches(module_names[4]) for f in filters), f"'{module_names[4]}' should NOT match"


def tests():
    # --- Existing tests ---
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
    assert empty.was_used()

    my_filters = '''
    up_blocks.0.(attentions|resnets|upsamplers).[02], up_blocks.1.*(conv|time.embed|attn2.to.v), up_blocks.2.resnets.[012].(conv|time)
    '''
    filters = [ModuleFilter(pattern, use_regex=True) for pattern in my_filters.strip().split(",")]
    assert filters[0].matches('up_blocks.0.attentions.0')
    assert filters[0].matches('up_blocks.0.resnets.2')
    assert not filters[0].matches('up_blocks.1.attentions.2')
    assert filters[1].matches('up_blocks.1.foo.bar.attn2.to.v')
    assert filters[2].matches('up_blocks.2.resnets.1.conv')

    # --- Call the new test function ---
    test_simple_substring_list()


def main():
    tests()
    print('All tests passed OK')


if __name__ == '__main__':
    main()
