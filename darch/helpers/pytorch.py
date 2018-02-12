from six import iteritems, itervalues
import darch.core as co
import torch.nn as nn


# NOTE: the init is very similar to the one for tensorflow. 
class PyTModule(co.Module):
    def __init__(self, name, name_to_h, compile_fn, 
            input_names, output_names, scope=None):
        co.Module.__init__(self, scope, name)

        if input_names is None:
            input_names = ["In"]
        for name in input_names:
            self._register_input(name)
        
        if output_names is None:
            output_names = ["Out"]        
        for name in output_names:
            self._register_output(name)

        for name, h in iteritems(name_to_h):
            self._register_hyperparameter(h, name)

        self._compile_fn = compile_fn

    def _compile(self):
        argnames = self._compile_fn.__code__.co_varnames

        kwargs = {}
        for name, h in iteritems(self.hs):
            kwargs[name] = h.get_val()
        for name, ix in iteritems(self.inputs):
            if name in argnames:        
                kwargs[name] = ix.val

        self._m = self._compile_fn(**kwargs)
        assert isinstance(self._m, nn.Module)

    def _forward(self):
        kwargs = {name : ix.val for name, ix in iteritems(self.inputs) }
        val = self._m(**kwargs)
        self.outputs['Out'].val = val  # TODO generalize for multiple outputs

# NOTE: perhaps have a simpler way of defining these things.

def _call_fn_on_torch_module(output_or_module_lst, fn):
    def fn_iter(x):
        d = vars(x)
        for v in itervalues(d):
            if isinstance(v, nn.Module):
                fn(v)
        return False

    module_lst = co.extract_unique_modules(output_or_module_lst)    
    co.backward_traverse(module_lst, fn_iter)

def get_pytorch_modules(output_or_module_lst):
    pyt_modules = set()
    _call_fn_on_torch_module(output_or_module_lst, lambda x: pyt_modules.add(x))
    return pyt_modules

def train(output_or_module_lst):
    _call_fn_on_torch_module(output_or_module_lst, lambda x: x.train())

def eval(output_or_module_lst):
    _call_fn_on_torch_module(output_or_module_lst, lambda x: x.eval())

def cuda(output_or_module_lst):
    _call_fn_on_torch_module(output_or_module_lst, lambda x: x.cuda())

def parameters(output_or_module_lst):
    ms = get_pytorch_modules(output_or_module_lst)
    ps = set()
    for m in ms:
        ps.update(m.parameters())
    return ps


# NOTE: how to adapt this is 
# NOTE: this is convenient to work with pytorch coming from darch. 
# typical functions are train, eval, cuda, parameters. 
# can use add_modules to support this.
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
