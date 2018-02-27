import graphviz
from six import itervalues, iteritems
import darch.core as co

def get_unconnected_inputs(output_lst):
    ix_lst = []
    def fn(x):
        for ix in itervalues(x.inputs):
            if not ix.is_connected():
                ix_lst.append(ix)
        return False

    co.traverse_backward(output_lst, fn)
    return ix_lst

def get_unconnected_outputs(input_lst):
    ox_lst = []
    def fn(x):
        for ox in itervalues(x.outputs):
            if not ox.is_connected():
                ox_lst.append(ox)
        return False

    co.traverse_forward(input_lst, fn)
    return ox_lst

def m2io(m):
    return (m.inputs, m.outputs)