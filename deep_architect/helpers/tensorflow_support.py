from __future__ import absolute_import
import deep_architect.core as co
import tensorflow as tf
import numpy as np


class TensorflowModule(co.Module):
    """Class for taking Tensorflow code and wrapping it in a DeepArchitect module.

    This class subclasses :class:`deep_architect.core.Module` as therefore inherits all
    the functionality associated to it (e.g., keeping track of inputs, outputs,
    and hyperparameters). It also enables to do the compile and forward
    operations for these types of modules once a module is fully specified,
    i.e., once all the hyperparameters have been chosen.

    The compile operation in this case creates all the variables used for the
    fragment of the computational graph associated to this module.
    The forward operation takes the variables that were created in the compile
    operation and constructs the actual computational graph fragment associated
    to this module.

    .. note::
        This module is abstract, meaning that it does not actually implement
        any particular Tensorflow computation. It simply wraps Tensorflow
        functionality in a DeepArchitect module. The instantiation of the Tensorflow
        variables is taken care by the `compile_fn` function that takes a two
        dictionaries, one of inputs and another one of outputs, and
        returns another function that takes a dictionary of inputs and creates
        the computational graph. This functionality makes extensive use of closures.

        The keys of the dictionaries that are passed to the compile
        and forward function match the names of the inputs and hyperparameters
        respectively. The dictionary returned by the forward function has keys
        equal to the names of the outputs.

    The usage of this class is best understood through examples.
    Examples for how to instantiate objects of this class to implement
    specific Tensorflow computation can be found in
    :mod:`deep_architect.search_spaces.tensorflow.conv2d`.

    Args:
        name (str): Name of the module
        name_to_hyperp (dict[str,deep_architect.core.Hyperparameter]): Dictionary of
            hyperparameters that the model depends on. The keys are the local
            names of the hyperparameters.
        compile_fn ((dict[str,object], dict[str,object]) -> ((dict[str,object]) -> ((dict[str,object]) | (dict[str,object], dict[tensorflow.python.framework.ops.Tensor,object], dict[tensorflow.python.framework.ops.Tensor,object])))):
            The first function takes two dictionaries with
            keys corresponding to `input_names` and `output_names` and returns
            a function that takes a dictionary with keys corresponding to
            `input_names` and returns a dictionary with keys corresponding
            to `output_names`. The first function may also return
            two additional dictionaries mapping Tensorflow placeholders to the
            values that they will take during training and test.
        input_names (list[str]): List of names for the inputs.
        output_names (list[str]): List of names for the outputs.
        scope (deep_architect.core.Scope, optional): Scope where the module will be
            registered.

    """

    def __init__(self,
                 name,
                 name_to_hyperp,
                 compile_fn,
                 input_names,
                 output_names,
                 scope=None):
        co.Module.__init__(self, scope, name)

        self._register(input_names, output_names, name_to_hyperp)
        self._compile_fn = compile_fn

    def _compile(self):
        input_name_to_val = self._get_input_values()
        hyperp_name_to_val = self._get_hyperp_values()

        out = self._compile_fn(input_name_to_val, hyperp_name_to_val)
        if isinstance(out, tuple):
            (self._fn, self.train_feed, self.eval_feed) = out
        else:
            self._fn = out

    def _forward(self):
        input_name_to_val = self._get_input_values()
        output_name_to_val = self._fn(input_name_to_val)
        self._set_output_values(output_name_to_val)

    def _update(self):
        pass


def siso_tensorflow_module(name, compile_fn, name_to_hyperp, scope=None):
    return TensorflowModule(name, name_to_hyperp, compile_fn, ['In'], ['Out'],
                            scope).get_io()


def siso_tensorflow_module_from_tensorflow_op_fn(layer_fn,
                                                 name_to_hyperp,
                                                 scope=None,
                                                 name=None):

    def compile_fn(di, dh):
        m = layer_fn(**dh)

        def forward_fn(di):
            return {"Out": m(di["In"])}

        return forward_fn

    if name is None:
        name = layer_fn.__name__

    return siso_tensorflow_module(name, compile_fn, name_to_hyperp, scope)


def get_feed_dicts(output_lst):
    """Get the training and evaluation dictionaries that map placeholders to the
    values that they should take during training and evaluation, respectively
    (e.g., used for dropout or batch normalization).

    Args:
        output_lst (list[deep_architect.core.Output]): List of outputs of the model
            (i.e., with no unspecified hyperparameters available) sampled
            from the search space.

    Returns:
        (dict, dict):
            Training and evaluation dictionaries where the keys are placeholders
            and the values are the values the placeholders should take during
            each of these stages.
    """
    train_feed = {}
    eval_feed = {}

    def fn(x):
        if hasattr(x, 'train_feed'):
            train_feed.update(x.train_feed)
        if hasattr(x, 'eval_feed'):
            eval_feed.update(x.eval_feed)
        return False

    co.traverse_backward(output_lst, fn)
    return (train_feed, eval_feed)


def get_num_trainable_parameters():
    return np.sum(
        [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
