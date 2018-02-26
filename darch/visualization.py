import matplotlib.pyplot as plt
import graphviz
from six import itervalues, iteritems
import darch.core as co
import numpy as np

def running_max(vs):
    return np.maximum.accumulate(vs)

def draw_graph(output_lst, draw_hyperparameters=False, 
        draw_io_labels=False, graph_name='graph', out_folderpath=None, 
        print_to_screen=True):
    assert print_to_screen or out_folderpath is not None

    g = graphviz.Digraph()
    edge_fs = '10'
    h_fs = '10'
    penwidth = '3'
    
    nodes = set()
    hs = set()
    def fn(m):
        nodes.add(m.get_name())
        for ix_localname, ix in iteritems(m.inputs):
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
                    ix.get_module().get_name())

        if draw_hyperparameters:
            for (h_localname, h) in iteritems(m.hyperps):
                hs.add(h.get_name())
                if not h.is_set():
                    label = h_localname
                else:
                    label = h_localname + '=' + str(h.val)

                g.edge(
                    h.get_name(), 
                    m.get_name(), 
                    label=label, fontsize=edge_fs)
        return False

    # generate most of the graph.
    module_lst = co.extract_unique_modules(output_lst)
    co.traverse_backward(output_lst, fn)

    # add the output terminals.
    for m in module_lst:
        for ox in itervalues(m.outputs):
                g.node(ox.get_name(), shape='house', penwidth=penwidth)
                g.edge(
                    ox.get_module().get_name(),
                    ox.get_name())
    
    # minor adjustments to attributes.
    for s in nodes:
        g.node(s, shape='invtrapezium', penwidth=penwidth)

    for s in hs:
        g.node(s, fontsize=h_fs)

    g.render(graph_name, out_folderpath, view=print_to_screen)  

class LinePlot:
    def __init__(self, title=None, xlabel=None, ylabel=None):
        self.data = []
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
    
    def add_line(self, xs, ys, label=None, err=None):
        d = {"xs" : xs, 
             "ys" : ys, 
             "label" : label,
             "err" : err}
        self.data.append(d)

    def plot(self, show=True, fpath=None):
        f = plt.figure()
        for d in self.data:
             plt.errorbar(d['xs'], d['ys'], yerr=d['err'], label=d['label'])
        
        if self.title is not None:
            plt.title(self.title)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        
        if any([d['label'] is not None for d in self.data]):
            plt.legend(loc='best')

        if fpath != None:
            f.savefig(fpath, bbox_inches='tight')
        if show:
            plt.show()
        return f