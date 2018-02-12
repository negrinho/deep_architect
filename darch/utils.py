import numpy as np
import graphviz
from six import itervalues, iteritems
import darch.core as co
from copy import deepcopy
import pickle

# TODO: may be worth to think about the right level of complexity for
# representing computation.
# add the dimensionality to all terminals.

# TODO: perhaps augment with different values.


# NOTE: may require to also talk about the other hyperparameters.
# NOTE: I should be able to write down the dimension in the input or output dim.
# this would be done upon instantiation; may change in dynamic frameworks.
def draw_graph(output_or_module_lst, draw_hyperparameters=False, 
        draw_io_labels=False, graph_name='graph', out_folderpath=None, 
        print_to_screen=True):
    assert print_to_screen or out_folderpath is not None

    g = graphviz.Digraph()
    edge_fs = '10'
    h_fs = '10'
    penwidth = '3'
    
    nodes = set()
    hs = set()
    def fn(x):
        nodes.add( x.get_name() )
        for ix_localname, ix in iteritems(x.inputs):
            if ix.is_connected():
                ox = ix.get_connected_output()
                if not draw_io_labels:
                    label = ''
                else:
                    for ox_iter_localname, ox_iter in iteritems(ox.get_module().outputs):
                        if ox_iter == ox:
                            ox_localname = ox_iter_localname
                            break
                    label = ix_localname + ':' + ox_localname

                g.edge(
                    ox.get_module().get_name(), 
                    ix.get_module().get_name(),
                    label=label, fontsize=edge_fs)

            # unconnected output
            else:
                g.node(ix.get_name(), shape='invhouse', penwidth=penwidth)
                g.edge(
                    ix.get_name(),
                    ix.get_module().get_name() )

        if draw_hyperparameters:
            for (h_localname, h) in iteritems(x.hs):
                hs.add( h.get_name() )
                if not h.is_set():
                    label = h_localname
                else:
                    label = h_localname + '=' + str(h.val)

                g.edge(
                    h.get_name(), 
                    x.get_name(), 
                    label=label, fontsize=edge_fs)

            ## TODO: if has to draw hyperparameters; 
            # do something about the hyperparameters that are not there.

        return False

    # generate most of the graph.
    module_lst = co.extract_unique_modules(output_or_module_lst)
    co.backward_traverse(module_lst, fn)

    # add the output terminals.
    for m in module_lst:
        for ox in itervalues(m.outputs):
                g.node(ox.get_name(), shape='house', penwidth=penwidth)
                g.edge(
                    ox.get_module().get_name(),
                    ox.get_name() )
    
    # minor adjustments to attributes.
    for s in nodes:
        g.node(s, shape='invtrapezium', penwidth=penwidth)

    for s in hs:
        g.node(s, fontsize=h_fs)

    g.render(graph_name, out_folderpath, view=print_to_screen)  

def copy_graph(output_or_module_lst):
    return deepcopy(output_or_module_lst)

def save_graph(output_or_module_lst, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(output_or_module_lst, f)    

def load_graph(fpath):
    with open(fpath, 'rb') as f:
        return pickle.load(f)

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

# TODO: needs to be done in all parts of the model. perhaps add more info.
def propagate_outputs(module):
    for ox in itervalues(module.outputs):
        for ix in ox.get_connected_inputs():
            ix.val = ox.val

def connect_sequentially(module_lst):
    for x, x_next in zip(module_lst[:-1], module_lst[1:]):
        x.outputs['Out'].connect(x_next.inputs['In'])

# TODO: can be extended for the case where there are multiple outputs.
def module_to_io(m):
    return (m.inputs, m.outputs) 

def module_lst_to_io(m_lst):
    return (m_lst[0].inputs, m_lst[-1].outputs)

def module_fn_to_io_fn(fn):
    return lambda : module_to_io( fn() )

def module_lst_fn_to_io_fn(fn):
    return lambda : module_lst_to_io( fn() )

def running_max(vs):
    return np.maximum.accumulate(vs)