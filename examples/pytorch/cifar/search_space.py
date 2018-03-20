import darch.helpers.pytorch as hpt
import darch.hyperparameters as hp
from model import CifarResNeXt


D = hp.Discrete


def SISOPTM(name, compile_fn, name_to_h={}, scope=None):
    return hpt.PyTModule(name, name_to_h, compile_fn,
            ['In'], ['Out'], scope)


def get_net(nlabels):
    def cfn(input_name_to_val, hyperp_name_to_val):
        net = CifarResNeXt(nlabels=nlabels, **hyperp_name_to_val)
        return lambda inp_dict: {'Out': net(inp_dict['In'])}, [net]
    return SISOPTM('CifarResNeXt', cfn, hyperparameters_fn())


def hyperparameters_fn():
    return {
        'cardinality': D([ 1, 2, 3 ]),
        'base_width': D([ 2, 4, 8 ]),
        'widen_factor': D([ 1, 2, 4 ]),
        'block_depth': D([ 0, 1, 2, 3, 4, 5 ]),
        'nr_stages': D([ 1, 2, 3, 4 ])
    }


def get_ss_fn(nlabels):
    def fn():
        mod = get_net(nlabels)
        return mod.inputs, mod.outputs, hyperparameters_fn()
    return fn
