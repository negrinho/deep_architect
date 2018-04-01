from six import iteritems
import darch.core as co
import torch.nn as nn

class PyTModule(co.Module):
    def __init__(self, name, name_to_hyperp, compile_fn,
            input_names, output_names, scope=None):
        """
        :type name: str
        :type name_to_hyperp: dict[str,darch.core.Hyperparameter]
        :type compile_fn: (dict[str,darch.core.Input], dict[str,darch.core.Hyperparameter]) ->
                          ((dict[str,darch.core.Input]) -> dict[str,darch.core.Output], list[nn.Module])
        :type input_names: list[str]
        :type output_names: list[str]
        :type scope: darch.core.Scope | None
        """
        co.Module.__init__(self, scope, name)
        self._register(input_names, output_names, name_to_hyperp)
        self._compile_fn = compile_fn

    def _compile(self):
        input_name_to_val = self._get_input_values()
        hyperp_name_to_val = self._get_hyperp_values()
        self._fn, self.pyth_modules = self._compile_fn(input_name_to_val, hyperp_name_to_val)
        for pyth_m in self.pyth_modules:
            assert isinstance(pyth_m, nn.Module)

    def _forward(self):
        input_name_to_val = self._get_input_values()
        output_name_to_val = self._fn(input_name_to_val)
        self._set_output_values(output_name_to_val)

# NOTE: this is done for the case where all the PyTorch modules are created
# using the helper here described, i.e., it assumes the existence of pyth_modules.
def _call_fn_on_pytorch_module(output_lst, fn):
    """
    :type output_lst: collections.Iterable[darch.core.Output]
    :type fn: (darch.core.Module) -> None
    """
    def fn_iter(mx):
        for pyth_m in mx.pyth_modules:
            fn(pyth_m)
        return False
    co.traverse_backward(output_lst, fn_iter)

def get_pytorch_modules(output_lst):
    all_modules = set()
    _call_fn_on_pytorch_module(output_lst, all_modules.add)
    return all_modules

def train(output_lst):
    """Applies :meth:`nn.Module.train` to all modules needed to compute the given outputs."""
    _call_fn_on_pytorch_module(output_lst, lambda pyth_m: pyth_m.train())

def eval(output_lst):
    """Applies :meth:`nn.Module.eval` to all modules needed to compute the given outputs."""
    _call_fn_on_pytorch_module(output_lst, lambda pyth_m: pyth_m.eval())

def cuda(output_lst):
    """Applies :meth:`nn.Module.cuda` to all modules needed to compute the given outputs."""
    _call_fn_on_pytorch_module(output_lst, lambda pyth_m: pyth_m.cuda())

def cpu(output_lst):
    """Applies :meth:`nn.Module.cpu` to all modules needed to compute the given outputs."""
    _call_fn_on_pytorch_module(output_lst, lambda pyth_m: pyth_m.cpu())

def parameters(output_lst):
    pyth_modules = get_pytorch_modules(output_lst)
    ps = set()
    for pyth_m in pyth_modules:
        ps.update(pyth_m.parameters())
    return ps

class PyTNetContainer(nn.Module):
    """
    Darch wrapper for pytorch modules.
    """
    def __init__(self, name_to_input, name_to_output):
        """
        :type name_to_input: dict[str,darch.core.Input]
        :type name_to_output: dict[str,darch.core.Output]
        """
        nn.Module.__init__(self)

        self.name_to_output = name_to_output
        self.name_to_input = name_to_input
        self._module_seq = None
        self._is_compiled = False

    def __call__(self, input_to_val):
        return self.forward(input_to_val)

    # TODO: needs additional error checking to make sure that the set of
    # outputs is correct.
    def forward(self, name_to_val):
        if self._module_seq is None:
            self._module_seq = co.determine_module_eval_seq(self.name_to_input.values())

        input_name_to_val = {ix: name_to_val[name]
            for name, ix in iteritems(self.name_to_input)}
        co.forward(input_name_to_val, self._module_seq)
        output_name_to_val = {name: ox.val
            for name, ox in iteritems(self.name_to_output)}

        if not self._is_compiled:
            modules = get_pytorch_modules(self.name_to_output.values())
            for i, m in enumerate(modules):
                self.add_module(str(i), m)
            self._is_compiled = True
        return output_name_to_val
