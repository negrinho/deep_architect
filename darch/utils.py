import graphviz
from six import itervalues, iteritems
import darch.core as co

def get_unconnected_inputs(output_or_module_lst):
    module_lst = co.extract_unique_modules(output_or_module_lst)

    ix_lst = []
    def fn(x):
        for ix in itervalues(x.inputs):
            if not ix.is_connected():
                ix_lst.append(ix)
        return False

    co.backward_traverse(module_lst, fn)
    return ix_lst

def get_unconnected_outputs(input_or_module_lst):
    module_lst = co.extract_unique_modules(input_or_module_lst)

    ox_lst = []
    def fn(x):
        for ox in itervalues(x.outputs):
            if not ox.is_connected():
                ox_lst.append(ox)
        return False

    co.forward_traverse(module_lst, fn)
    return ox_lst

def connect_sequentially(module_lst):
    for x, x_next in zip(module_lst[:-1], module_lst[1:]):
        x.outputs['Out'].connect(x_next.inputs['In'])