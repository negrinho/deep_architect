import darch.helpers.pytorch as htorch
import darch.hyperparameters as hp

# Alias discrete hyperparameters for some reason?
D = hp.Discrete

def siso_torchm(name, compile_fn, name_to_h, scope=None):
    """
    Name:
    comile_fn:
    name_to_h:
    scope (default=None):
    """
    return htorch.PyTModule(name, name_to_h, ['In'], ['Out'], scope).get_io()