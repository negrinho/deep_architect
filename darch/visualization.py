import graphviz
from six import itervalues, iteritems
import darch.core as co

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
