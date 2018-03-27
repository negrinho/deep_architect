import torch.nn as nn


class EmptyNet(nn.Module):
    def __init__(self):
        super(EmptyNet, self).__init__()

    def forward(self, x):
        return x


def get_empty_cfn():
    def cfn(_, __):
        net = EmptyNet()
        return lambda inp_dict: {'Out': net(inp_dict['In'])}, [net]
    return cfn


def test_pytmodule():
    from darch.helpers.pytorch import PyTModule

    cfn = get_empty_cfn()
    mod = PyTModule(name='test',
                    name_to_hyperp={},
                    compile_fn=cfn,
                    input_names=['In'],
                    output_names=['Out'],
                    scope=None)

    arbitrary_value = 'arbitrary value'
    mod.inputs['In'].val = arbitrary_value

    assert not mod._is_compiled
    mod.forward()
    assert mod._is_compiled
