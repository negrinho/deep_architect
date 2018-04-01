from darch import core as co, modules as mo
from darch.contrib.search_spaces.pytorch.common import siso_torchm, D
import numpy as np
import torch
import tensorflow as tf

# Define a search space for Deep Neural Net architectures and hyperparameters.

# Define Initializers

def constant_initializer(constant):
    """
    constant: scalar value to set all values to
    returns: fn shape -> tensor
    """
    pass
        

# TODO: Implement the rest as in contrib.tensorflow.dnn