from __future__ import absolute_import
import darch.core as co
import tensorflow as tf
import numpy as np


class TFModule(co.Module):
    """
    Tensorflow wrapper for darch modules.
    """
    def __init__(self, name, name_to_hyperp, compile_fn,
            input_names, output_names, scope=None):
        """
        :type name: str
        :type name_to_hyperp: dict[str,darch.core.Hyperparameter]
        :type compile_fn: (dict[str,darch.core.Input], dict[str,darch.core.Hyperparameter]) -> (
                          ((dict[str,darch.core.Input]) -> dict[str,darch.core.Output],
                           dict[tensorflow.python.framework.ops.Tensor,Any],
                           dict[tensorflow.python.framework.ops.Tensor,Any]) |
                          (dict[str,darch.core.Input]) -> dict[str,darch.core.Output])
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

        out = self._compile_fn(input_name_to_val, hyperp_name_to_val)
        if isinstance(out, tuple):
            (self._fn, self.train_feed, self.eval_feed) = out
        else:
            self._fn = out

    def _forward(self):
        input_name_to_val = self._get_input_values()
        output_name_to_val = self._fn(input_name_to_val)
        self._set_output_values(output_name_to_val)

    def update(self):
        pass


def get_feed_dicts(output_lst):
    """
    :type output_lst: collections.Iterable[darch.core.Output]
    :rtype: (dict[tensorflow.python.framework.ops.Tensor,Any], dict[tensorflow.python.framework.ops.Tensor,Any])
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
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
