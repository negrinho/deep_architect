import deep_architect.helpers.tensorflow_support as htf
import deep_architect.hyperparameters as hp

D = hp.Discrete


def siso_tensorflow_module(name, compile_fn, name_to_hyperp, scope=None):
    return htf.TensorflowModule(name, compile_fn, name_to_hyperp, ['in'],
                                ['out'], scope).get_io()
