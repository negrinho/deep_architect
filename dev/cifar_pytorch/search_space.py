import deep_architect.helpers.pytorch as hpt
import deep_architect.hyperparameters as hp
from model import CifarResNeXt
from deep_architect.contrib.misc.search_spaces.pytorch.common import siso_pytorch_module


D = hp.Discrete


def SISOPTM(name, compile_fn, name_to_h={}, scope=None):
    return hpt.PyTorchModule(name, name_to_h, compile_fn,
            ['In'], ['Out'], scope)


def get_net(nlabels):
    def cfn(di, dh):
        net = CifarResNeXt(nlabels=nlabels, **dh)
        def fn(di):
            return lambda inp_dict: {'Out': net(di['In'])}, [net]
    return siso_pytorch_module('CifarResNeXt', cfn, hyperparameters_fn())

def hyperparameters_fn():
    return {
        'cardinality': D([1, 2, 3]),
        'base_width': D([2, 4, 8]),
        'widen_factor': D([1, 2, 4]),
        'block_depth': D([0, 1, 2, 3, 4, 5]),
        'nr_stages': D([1, 2, 3, 4])
    }


def get_ss_fn(nlabels):
    def fn():
        mod = get_net(nlabels)
        return mod.inputs, mod.outputs, {}
    return fn
