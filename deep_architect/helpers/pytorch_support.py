from six import iteritems
import torch.nn as nn
import deep_architect.core as co
from deep_architect.hyperparameters import D


class PyTorchModule(co.Module):
    """Class for taking Pytorch code and wrapping it in a DeepArchitect module.

    This class subclasses :class:`deep_architect.core.Module` as therefore inherits all
    the information associated to it (e.g., inputs, outputs, and hyperparameters).
    It also enables to do the compile and forward operations for these types of
    modules once a module is fully specified, i.e., once all the hyperparameters
    have been chosen.

    The compile operation in this case instantiates any Pytorch modules necessary
    for the computation associated to this module.
    The forward operation takes the variables that were created in the compile
    operation and constructs the actual computational graph fragment associated
    to this module.

    See :class:`deep_architect.helpers.tensorflow_support.TensorflowModule` for a similar class for
    Tensorflow. One of the main differences is that Tensorflow deals with
    static computational graphs, so the forward functionality is often only
    called once per creation for the graph creation. Pytorch requires calling
    forward for each tensor of data that is fed through the network.

    .. note::
        This module is abstract, meaning that it does not actually implement
        any particular Pytorch computation. It simply wraps Pytorch
        functionality in a DeepArchitect module. This functionality makes extensive use
        of closures.

        The keys of the dictionaries that are passed to the compile function
        match the names of the inputs and hyperparameters, respectively.
        The keys of the dictionary that are passed to the forward function match
        the names of the inputs. The keys of dictionary returned by the forward
        function match the names of the outputs.

    The usage of this class is best understood through examples.
    Examples for how to instantiate objects of this class to implement
    specific Pytorch computations can be found in
    :mod:`deep_architect.search_spaces.pytorch.conv2d`.

    Args:
        name (str): Name of the module
        compile_fn ((dict[str,object], dict[str,object]) -> ((dict[str,object]) -> (dict[str,object], list[torch.nn.Modules]))):
            The first function takes two dictionaries with
            keys corresponding to `input_names` and `output_names` and returns
            a function that takes a dictionary with keys corresponding to
            `input_names` and returns a dictionary with keys corresponding
            to `output_names` and a list of Pytorch modules involved in the
            computation of the DeepArchitect module.
        name_to_hyperp (dict[str,deep_architect.core.Hyperparameter]): Dictionary of
            hyperparameters that the model depends on. The keys are the local
            names of the hyperparameters.
        input_names (list[str]): List of names for the inputs.
        output_names (list[str]): List of names for the outputs.
        scope (deep_architect.core.Scope, optional): Scope where the module will be
            registered.
    """

    def __init__(self,
                 name,
                 compile_fn,
                 name_to_hyperp,
                 input_names,
                 output_names,
                 scope=None):
        co.Module.__init__(self, scope, name)
        hyperparam_dict = {}
        for h in name_to_hyperp:
            if not isinstance(name_to_hyperp[h], co.Hyperparameter):
                hyperparam_dict[h] = D([name_to_hyperp[h]])
            else:
                hyperparam_dict[h] = name_to_hyperp[h]

        self._register(input_names, output_names, hyperparam_dict)
        self._compile_fn = compile_fn

    def _compile(self):
        input_name_to_val = self._get_input_values()
        hyperp_name_to_val = self._get_hyperp_values()
        self._fn, self.pyth_modules = self._compile_fn(input_name_to_val,
                                                       hyperp_name_to_val)
        for pyth_m in self.pyth_modules:
            assert isinstance(pyth_m, nn.Module)

    def _forward(self):
        input_name_to_val = self._get_input_values()
        output_name_to_val = self._fn(input_name_to_val)
        self._set_output_values(output_name_to_val)

    def _update(self):
        pass


def siso_pytorch_module(name, compile_fn, name_to_hyperp, scope=None):
    return PyTorchModule(name, compile_fn, name_to_hyperp, ['In'], ['Out'],
                         scope).get_io()


def siso_pytorch_module_from_pytorch_layer_fn(layer_fn,
                                              name_to_hyperp,
                                              scope=None,
                                              name=None):

    def compile_fn(di, dh):
        m = layer_fn(**dh)

        def forward_fn(di):
            return {"Out": m(di["In"])}

        return forward_fn, [m]

    if name is None:
        name = layer_fn.__name__

    return siso_pytorch_module(name, compile_fn, name_to_hyperp, scope)


# NOTE: this is done for the case where all the PyTorch modules are created
# using the helper here described, i.e., it assumes the existence of pyth_modules.
def _call_fn_on_pytorch_module(outputs, fn):

    def fn_iter(mx):
        if hasattr(mx, 'pyth_modules'):
            for pyth_m in mx.pyth_modules:
                fn(pyth_m)
        return False

    co.traverse_backward(outputs, fn_iter)


def get_pytorch_modules(outputs):
    all_modules = set()
    _call_fn_on_pytorch_module(outputs, all_modules.add)
    return all_modules


def train(outputs):
    """Applies :meth:`torch.nn.Module.train` to all modules needed to compute the outputs."""
    _call_fn_on_pytorch_module(outputs, lambda pyth_m: pyth_m.train())


def eval(outputs):
    """Applies :meth:`torch.nn.Module.eval` to all modules needed to compute the outputs."""
    _call_fn_on_pytorch_module(outputs, lambda pyth_m: pyth_m.eval())


# TODO: this needs to be changed.
def cuda(outputs, *args, **kwargs):
    _call_fn_on_pytorch_module(
        outputs, lambda pyth_m: pyth_m.cuda(*args, **kwargs))


def cpu(outputs):
    """Applies :meth:`torch.nn.Module.cpu` to all modules needed to compute the outputs."""
    _call_fn_on_pytorch_module(outputs, lambda pyth_m: pyth_m.cpu())


def parameters(outputs):
    pyth_modules = get_pytorch_modules(outputs)
    ps = set()
    for pyth_m in pyth_modules:
        ps.update(pyth_m.parameters())
    return ps


class PyTorchModel(nn.Module):
    """Encapsulates a network of modules of type :class:`deep_architect.helpers.pytorch_support.PyTorchModule`
    in a way that they can be used as :class:`torch.nn.Module`, e.g.,
    functionality to move the computation of the GPU or to get all the parameters
    involved in the computation are available.

    Using this class is the recommended way of wrapping a Pytorch architecture
    sampled from a search space. The topological order for evaluating for
    doing the forward computation of the architecture is computed by the
    container and cached for future calls to forward.

    Args:
        inputs (dict[str,deep_architect.core.Input]): Dictionary of names to inputs.
        outputs (dict[str,deep_architect.core.Output]): Dictionary of names to outputs.
    """

    def __init__(self, inputs, outputs):
        nn.Module.__init__(self)

        self.outputs = outputs
        self.inputs = inputs
        self._module_seq = None
        self._is_compiled = False

    def __call__(self, input_name_to_val):
        return self.forward(input_name_to_val)

    # TODO: needs additional error checking to make sure that the set of
    # outputs is correct.
    def forward(self, input_name_to_val):
        """Forward computation of the module that is represented through the
        graph of DeepArchitect modules.
        """
        if self._module_seq is None:
            self._module_seq = co.determine_module_eval_seq(
                self.inputs.values())

        input_to_val = {
            ix: input_name_to_val[name] for name, ix in iteritems(self.inputs)
        }
        co.forward(input_to_val, self._module_seq)
        output_name_to_val = {
            name: ox.val for name, ox in iteritems(self.outputs)
        }

        if not self._is_compiled:
            modules = get_pytorch_modules(self.outputs)
            for i, m in enumerate(modules):
                self.add_module(str(i), m)
            self._is_compiled = True
        return output_name_to_val