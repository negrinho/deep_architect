
import darch.helpers.tensorflow as htf
import darch.hyperparameters as hp

D = hp.Discrete

def siso_tfm(name, compile_fn, name_to_h, scope=None):
    return htf.TFModule(name, name_to_h, compile_fn, ['In'], ['Out'], scope).get_io()