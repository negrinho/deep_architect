from __future__ import absolute_import
from six import iteritems
import tensorflow as tf
import darch.core as co
import darch.modules as mo


# class TFSISOModule(mo.SISOModule):
#     def __init__(self, name, name_to_h, compile_fn, scope=None):
#         mo.SISOModule.__init__(self, scope, name)

#         for name, h in iteritems(name_to_h):
#             self._register_hyperparameter(h, name)
#         self._compile_fn = compile_fn

#     def _compile(self):
#         argnames = self._compile_fn.__code__.co_varnames

#         kwargs = {}
#         for name, h in iteritems(self.hs):
#             if name in argnames:
#                 kwargs[name] = h.get_val()
#         for name, ix in iteritems(self.inputs):
#             if name in argnames:        
#                 kwargs[name] = ix.val

#         out = self._compile_fn(**kwargs)
#         if isinstance(out, tuple):
#             (self._fn, self.train_feed, self.eval_feed) = out
#         else:
#             self._fn = out

#     def _forward(self):
#         self.outputs['Out'].val = self._fn(self.inputs['In'].val)

# NOTE: most of this is from the other module. 
# do a static module.
class TFModule(co.Module):
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

    # NOTE: while it may not depend on the inputs, it should 
    # always depend on the hyperparameters, otherwise it does not 
    # make sense.
    def _compile(self):
        argnames = self._compile_fn.__code__.co_varnames

        kwargs = {}
        for name, h in iteritems(self.hs):
            kwargs[name] = h.get_val()
        for name, ix in iteritems(self.inputs):
            if name in argnames:        
                kwargs[name] = ix.val

        out = self._compile_fn(**kwargs)
        if isinstance(out, tuple):
            (self._fn, self.train_feed, self.eval_feed) = out
        else:
            self._fn = out

    def _forward(self):
        kwargs = {name : ix.val for name, ix in iteritems(self.inputs) }
        name_to_val = self._fn(**kwargs)
        for name, val in iteritems(name_to_val):
            self.outputs[name].val = val

# class TFSISOModule(TFModule):
#     def __init__(self, name, name_to_h, compile_fn, scope=None):
#         co.Module.__init__(self, name, name_to_h, compile_fn, scope=scope)

#     def _forward(self):
#         self.outputs['Out'].val = self._fn(self.inputs['In'].val)

def get_feed_dicts(output_or_module_lst):
    train_feed = {}
    eval_feed = {}
    def fn(x):
        if hasattr(x, 'train_feed'):
            train_feed.update(x.train_feed)
        if hasattr(x, 'eval_feed'):
            eval_feed.update(x.eval_feed)
        return False

    module_lst = co.extract_unique_modules(output_or_module_lst)    
    co.backward_traverse(module_lst, fn)

    return (train_feed, eval_feed)

# NOTE: it is good to have some example functions here.

### NOTE: this is for op visualization
# useful for debugging of tensorflow code and of darch code.
# TODO: add graphical visualization capabilities.
# def parents(op):
#   return set(input.op for input in op.inputs)

# def children(op):
#   return set(op for out in op.outputs for op in out.consumers())

# def get_graph():
#   """Creates dictionary {node: {child1, child2, ..},..} for current
#   TensorFlow graph. Result is compatible with networkx/toposort"""

#   ops = tf.get_default_graph().get_operations()
#   return {op: children(op) for op in ops}

# def print_tf_graph(graph):
#   """Prints tensorflow graph in dictionary form."""
#   for node in graph:
#     for child in graph[node]:
#       print("%s -> %s" % (node.name, child.name))

# TODO: alternatively, just have a single one with all the helpers for all cases.
# it can be inherited.