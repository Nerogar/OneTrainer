import re


class ModuleFilter:
    def __init__(self, pattern):
        assert pattern.isprintable(), f'Custom layer filter contains non-printable characters: {repr(pattern)}'
        pattern = pattern.strip()
        # empty patterns *are* allowed and will match all layers, resulting in a full training.

        self.__used = False
        self._pattern = pattern

    def __repr__(self):
        return repr(self._pattern)

    def matches(self, module_name):
        if re.search(self._pattern, module_name):
            self.__used = True
            return True

        return False

    def was_used(self):
        return self.__used

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

    asterisk = ModuleFilter('.*')
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

    f = ModuleFilter('up_blocks.0.attentions.2.transformer_blocks.[56].attn[12].to_v')
    assert f.matches('lora.unet.up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_v')
    assert f.matches('lora.unet.up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_v')
    assert not f.matches('lora.unet.up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_k')
    assert not f.matches('lora.unet.up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_v')
    assert f.was_used()

    # layer lora_unet_down_blocks_2_attentions_0_transformer_blocks_4_attn2_to_v.lora_down.weight has norm 2.558
    f = ModuleFilter('down_blocks.2.attentions.0.transformer_blocks.[468].attn2.to_v')
    assert not f.matches('down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_v')
    assert not f.was_used()
    assert f.matches('down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_v')
    assert f.was_used()

    try:
        f = ModuleFilter('invalid_regex_(foo|bar')
        f.matches('raises_error')
        assert False, 'should have raised an error'
    except re.PatternError:
        pass

    f = ModuleFilter('up_bl..ks')
    assert f.matches('up_blocks.0.attentions.1')

    f = ModuleFilter('down_blocks.2.attentions.[01].transformer_blocks.[0-9].ff.net.0.proj')
    assert f.matches('lora.unet.down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj')
    assert f.matches('lora.unet.down_blocks.2.attentions.1.transformer_blocks.6.ff.net.0.proj')

    empty = ModuleFilter('')
    for name in block_names:
        assert empty.matches(name)

    my_filters = '''
    up_blocks.0.(attentions|resnets|upsamplers).[02], up_blocks.1.*(conv|time.embed|attn2.to.v), up_blocks.2.resnets.[012].(conv|time)
    '''
    filters = [ModuleFilter(pattern) for pattern in my_filters.strip().split(",")]
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
