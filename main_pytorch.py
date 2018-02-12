from __future__ import print_function
import darch.core as co
import darch.hyperparameters as hps
import darch.utils as ut
import darch.pytorch_helpers as pyt_helpers
import darch.searchers as se
import darch.modules as mo
import torch
import torch.nn as nn
import numpy as np

M = pyt_helpers.PyTModule
D = hps.Discrete

def SISOPYTM(name, compile_fn, name_to_h={}, scope=None):
    return pyt_helpers.PyTModule(name, name_to_h, compile_fn, 
            ['In'], ['Out'], scope)

## TODO: search space inspired by LSTM cell design.
# i = \mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
# f = \mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
# g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
# o = \mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
# c' = f * c + i * g \\
# h' = o * \tanh(c') \\

# i = \mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\


# class ModuleWrapper(nn.Module):
#     def __init__(self, fn, **kwargs):
#         nn.Module.__init__(self)
#         self.m = fn(**kwargs)
#     def forward(self, **kwargs):
#         return {'Out' : self.m(**kwargs)}

# def ReLU():
#     return M("ReLU", {}, lambda: ModuleWrapper(lambda: nn.ReLU), ['input'], ['Out'])
    

def ReLU():
    class Mod(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.m = nn.ReLU()
        def forward(self, In):
            return {'Out' : self.m(In)}

    return SISOPYTM('ReLU', lambda: Mod(), {})




# needs a combine for input and hidden
# needs a combine for input and forget
# needs a combine for 

# i = \mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
# f = \mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
# g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
# o = \mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
# c' = f * c + i * g \\
# h' = o * \tanh(c') \\

# input_size, hidden_size

# inputs are input, hidden 
# outputs are hidden and context.


# def SM(name, compile_fn, name_to_h={}, scope=None):
#     return M(name, name_to_h, compile_fn, ['In'], ['Out'], scope)

if __name__ == '__main__':
    
    m = ReLU()

    mw = pyt_helpers.PyTNetContainer(m.inputs, m.outputs)

    x = torch.randn(10)
    print(mw({'In' : x}))
