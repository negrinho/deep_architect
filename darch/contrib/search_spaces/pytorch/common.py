import darch.helpers.pytorch as hpt
import darch.hyperparameters as hp

# Alias discrete hyperparameters for brevity. s
D = hp.Discrete

def siso_torchm(name, compile_fn, name_to_hyperp, scope=None):
    return hpt.PyTModule(name, name_to_hyperp, compile_fn, ['In'], ['Out'], scope).get_io()