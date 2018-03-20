import torch.nn as nn


def get_empty_net():
    def cfn(input_name_to_cal, hyperp_name_to_val):
        class Net(nn.Module):
            def forward(self, x):
                return x
        net = Net()
        return lambda inp_dict: {'Out': net(inp_dict['In'])}, [net]
    return cfn


def test_pytmodule():
    from darch.helpers.pytorch import PyTModule

    cfn = get_empty_net()
    mod = PyTModule(name='test',
                    name_to_hyperp={},
                    compile_fn=cfn,
                    input_names=['In'],
                    output_names=['Out'],
                    scope=None)

    arbitray_value = 'arbitrary value'
    mod.inputs['In'].val = arbitray_value

    assert not mod._is_compiled
    mod.forward()
    assert mod._is_compiled
