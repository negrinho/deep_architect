import darch.helpers.pytorch as hpt
import darch.hyperparameters as hp
from examples.pytorch.cifar.model import CifarResNeXt


D = hp.Discrete


def SISOPTM(name, compile_fn, name_to_h={}, scope=None):
    return hpt.PyTModule(name, name_to_h, compile_fn,
            ['In'], ['Out'], scope)


def get_net(nlabels):
    def cfn(cardinality, block_depth, base_width, nr_stages, widen_factor):
        # def fn(In):
        net = CifarResNeXt(cardinality, block_depth, nlabels, base_width, nr_stages, widen_factor)
            # return {'Out': net(In)}
        return net
        # return fn
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