from six import iteritems, itervalues
import darch.core as co
import torch.nn as nn

class PyTModule(co.Module):
    def __init__(self, name, name_to_hyperp, compile_fn, 
            input_names, output_names, scope=None):
        co.Module.__init__(self, scope, name)
        self._register(input_names, output_names, name_to_hyperp)
        self._compile_fn = compile_fn

    def _compile(self):
        input_name_to_val = self._get_input_values()
        hyperp_name_to_val = self._get_hyperp_values()
        self._m = self._compile_fn(input_name_to_val, hyperp_name_to_val)
        assert isinstance(self._m, nn.Module)

    def _forward(self):
        kwargs = {name : ix.val for name, ix in iteritems(self.inputs) }
        val = self._m(**kwargs)
        self.outputs['Out'].val = val  # TODO generalize for multiple outputs

def _call_fn_on_torch_module(output_lst, fn):
    def fn_iter(x):
        d = vars(x)
        for v in itervalues(d):
            if isinstance(v, nn.Module):
                fn(v)
        return False
    co.traverse_backward(output_lst, fn_iter)

def get_pytorch_modules(output_lst):
    pyt_modules = set()
    _call_fn_on_torch_module(output_lst, lambda x: pyt_modules.add(x))
    return pyt_modules

def train(output_lst):
    _call_fn_on_torch_module(output_lst, lambda x: x.train())

def eval(output_lst):
    _call_fn_on_torch_module(output_lst, lambda x: x.eval())

def cuda(output_lst):
    _call_fn_on_torch_module(output_lst, lambda x: x.cuda())

def parameters(output_lst):
    ms = get_pytorch_modules(output_lst)
    ps = set()
    for m in ms:
        ps.update(m.parameters())
    return ps

class PyTNetContainer(nn.Module):
    def __init__(self, name_to_input, name_to_output):
        nn.Module.__init__(self)        

        self.name_to_output = name_to_output
        self.name_to_input = name_to_input
        self._module_seq = None
        self._is_compiled = False
    
    def __call__(self, input_to_val):
        return self.forward(input_to_val)

    def forward(self, name_to_val):
        if self._module_seq is None:
            self._module_seq = co.determine_module_eval_seq_general(
                self.name_to_input.values(), self.name_to_output.values())[0]

        input_to_val = {ix : name_to_val[name] for name, ix in iteritems(self.name_to_input)}
        co.forward(input_to_val, self._module_seq)
        d = {name : ox.val for name, ox in iteritems(self.name_to_output)}

        if not self._is_compiled:
            modules = get_pytorch_modules(self.name_to_output.values())
            for i, m in enumerate(modules):
                self.add_module(str(i), m)
            self._is_compiled = True
        return d