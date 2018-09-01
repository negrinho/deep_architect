import deep_architect.helpers.pytorch as hpt
import deep_architect.hyperparameters as hp

# Alias discrete hyperparameters for brevity. s
D = hp.Discrete

def siso_pytorch_module(name, compile_fn, name_to_hyperp, scope=None):
    return hpt.PyTorchModule(name, name_to_hyperp, compile_fn, ['In'], ['Out'], scope).get_io()