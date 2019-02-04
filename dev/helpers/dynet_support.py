import dynet as dy
import deep_architect.modules as mo
import deep_architect.core as co
import deep_architect.hyperparameters as hp


class DyParameterCollection(dy.ParameterCollection):
    """Wrapper around dynet ParameterCollection object

    DyNet ParameterCollection is an object that holds the parameters of the model.
    DyNet Module needs this to be able to add parameters into the model (using
    dynet.ParameterCollection.add_parameters() or .add_lookup_parameters()). This
    wrapper object needs only to be created once, import to search space modules
    to add parameters, and renew every time a new architecture is sampled.

    """

    def __init__(self):
        self.params = dy.ParameterCollection()

    def renew_collection(self):
        """Renew every time new architecture is sampled to clear out old parameters"""
        self.params = dy.ParameterCollection()

    def get_collection(self):
        """Call inside a search space module, and when needs to add to trainer"""
        return self.params


class DyNetModule(co.Module):
    """Wrapper Module, see TFModule for TensorFlow or PyTModule for PyTorch"""

    def __init__(self,
                 name,
                 compile_fn,
                 name_to_hyperp,
                 input_names,
                 output_names,
                 scope=None):
        co.Module.__init__(self, scope, name)
        self._register(input_names, output_names, name_to_hyperp)
        self._compile_fn = compile_fn

    def _compile(self):
        input_name_to_val = self._get_input_values()
        hyperp_name_to_val = self._get_hyperp_values()

        self._fn = self._compile_fn(input_name_to_val, hyperp_name_to_val)

    def _forward(self):
        input_name_to_val = self._get_input_values()
        output_name_to_val = self._fn(input_name_to_val)
        self._set_output_values(output_name_to_val)

    def _update(self):
        pass


def siso_dynet_module(name, compile_fn, name_to_hyperp, scope=None):
    "Dynet module for single-input, single-output"
    return DyNetModule(name, compile_fn, name_to_hyperp, ['In'], ['Out'],
                       scope).get_io()
