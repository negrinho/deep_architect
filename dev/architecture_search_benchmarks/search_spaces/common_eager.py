import darch.hyperparameters as hp
from dev.architecture_search_benchmarks.helpers import tfeager as htfe

D = hp.Discrete

def siso_tfem(name, compile_fn, name_to_hyperp, scope=None):
    return htfe.TFEModule(name, name_to_hyperp, compile_fn, ['In'], ['Out'], scope).get_io()