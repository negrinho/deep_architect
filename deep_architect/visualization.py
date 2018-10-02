import matplotlib.pyplot as plt
import graphviz
from six import itervalues, iteritems
import deep_architect.core as co
import deep_architect.utils as ut
import deep_architect.searchers as se
import numpy as np


def running_max(vs):
    return np.maximum.accumulate(vs)


def draw_graph(output_lst,
               draw_hyperparameters=True,
               draw_io_labels=True,
               draw_module_hyperparameter_info=True,
               out_folderpath=None,
               graph_name='graph',
               print_to_screen=True):
    """Draws a graph representation of the current state of the search space.

    All edges are directed. An edge between two modules represents the output of
    the first module going into the input of the second module. An edge between
    a module and an hyperparameter represents the dependency of that module on
    that hyperparameter.

    This visualization functionality is useful to verify that the description of
    the search space done through the domain-specific language encodes the
    intended search space.

    .. note::
        All drawing options are set to `True` to give a sense of all the
        information that can be displayed simultaneously. This can lead to
        cluttered graphs and slow rendering. We recommend changing the defaults
        according to the desired behavior.

        For generating a representation for a fully specified model,
        we recommend setting `draw_hyperparameters` to `False`, as if we are
        only concerned with the resulting model, the sharing structure of
        hyperparameters does not really matter. We recommend using
        `draw_hyperparameters` set to `True` when the user wishes to visualize
        the hyperparameter sharing pattern, e.g., to verify that it has been
        implemented correctly.

    Args:
        output_lst (list[deep_architect.core.Output]): List of outputs from which we
            can reach all the modules in the search space by backwards traversal.
        draw_hyperparameters (bool, optional): Draw hyperparameter nodes in the
            graph, representing the dependencies between hyperparameters and
            modules.
        draw_io_labels (bool, optional): If `True`,
            draw edge labels connecting different modules
            with the local names of the input and output that are being connected.
        draw_module_hyperparameter_info (bool, optional): If `True`,
            draw the hyperparameters of a module alongside the module.
        graph_name (str, optional): Name of the file used to store the
            rendered graph. Only needs to be provided if we desire to output
            the graph to a file, rather than just show it.
        out_folderpath (str, optional): Folder to which to store the PDF file with
            the rendered graph. If no path is provided, no file is created.
        print_to_screen (bool, optional): Shows the result of rendering the
            graph directly to screen.
    """
    assert print_to_screen or out_folderpath is not None

    g = graphviz.Digraph()
    edge_fs = '10'
    h_fs = '10'
    penwidth = '1'

    def _draw_connected_input(ix_localname, ix):
        ox = ix.get_connected_output()
        if not draw_io_labels:
            label = ''
        else:
            ox_localname = None
            for ox_iter_localname, ox_iter in iteritems(
                    ox.get_module().outputs):
                if ox_iter == ox:
                    ox_localname = ox_iter_localname
                    break
            assert ox_localname is not None
            label = ix_localname + ':' + ox_localname

        g.edge(
            ox.get_module().get_name(),
            ix.get_module().get_name(),
            label=label,
            fontsize=edge_fs)

    def _draw_unconnected_input(ix_localname, ix):
        g.node(ix.get_name(), shape='invhouse', penwidth=penwidth)
        g.edge(ix.get_name(), ix.get_module().get_name())

    def _draw_module_hyperparameter(m, h_localname, h):
        if h.has_value_assigned():
            label = h_localname + '=' + str(h.get_value())
        else:
            label = h_localname

        g.edge(h.get_name(), m.get_name(), label=label, fontsize=edge_fs)

    def _draw_dependent_hyperparameter_relations(h_dep):
        for h_localname, h in iteritems(h_dep._hyperps):
            if h.has_value_assigned():
                label = h_localname + '=' + str(h.get_value())
            else:
                label = h_localname

            g.edge(
                h.get_name(), h_dep.get_name(), label=label, fontsize=edge_fs)

    def _draw_module_hyperparameter_info(m):
        g.node(
            m.get_name(),
            xlabel="<" + '<br align="right"/>'.join([
                '<FONT POINT-SIZE="%s">' % h_fs + h_localname +
                ('=' + str(h.get_value()) if h.has_value_assigned() else '') +
                "</FONT>" for h_localname, h in iteritems(m.hyperps)
            ]) + ">")

    def _draw_output_terminal(ox_localname, ox):
        g.node(ox.get_name(), shape='house', penwidth=penwidth)
        g.edge(ox.get_module().get_name(), ox.get_name())

    nodes = set()

    def fn(m):
        """Adds the module information to the graph that is local to the module.
        """
        nodes.add(m.get_name())
        for ix_localname, ix in iteritems(m.inputs):
            if ix.is_connected():
                _draw_connected_input(ix_localname, ix)
            else:
                _draw_unconnected_input(ix_localname, ix)

        if draw_hyperparameters:
            for h_localname, h in iteritems(m.hyperps):
                _draw_module_hyperparameter(m, h_localname, h)

        if draw_module_hyperparameter_info:
            _draw_module_hyperparameter_info(m)
        return False

    # generate most of the graph.
    co.traverse_backward(output_lst, fn)

    # drawing the hyperparameter graph.
    if draw_hyperparameters:
        hs = co.get_all_hyperparameters(output_lst)

        for h in hs:
            if isinstance(h, co.DependentHyperparameter):
                _draw_dependent_hyperparameter_relations(h)

            g.node(h.get_name(), fontsize=h_fs)

    # add the output terminals.
    for m in co.extract_unique_modules(output_lst):
        for ox_localname, ox in iteritems(m.outputs):
            _draw_output_terminal(ox_localname, ox)

    # minor adjustments to attributes.
    for s in nodes:
        g.node(s, shape='rectangle', penwidth=penwidth)

    if print_to_screen or out_folderpath is not None:
        g.render(graph_name, out_folderpath, view=print_to_screen, cleanup=True)


def draw_graph_evolution(output_lst,
                         hyperp_value_lst,
                         out_folderpath,
                         graph_name='graph',
                         draw_hyperparameters=True,
                         draw_io_labels=True,
                         draw_module_hyperparameter_info=True):

    def draw_fn(i):
        return draw_graph(
            output_lst,
            draw_hyperparameters=draw_hyperparameters,
            draw_io_labels=draw_io_labels,
            draw_module_hyperparameter_info=draw_module_hyperparameter_info,
            out_folderpath=out_folderpath,
            graph_name=graph_name + '-%d' % i,
            print_to_screen=False)

    draw_fn(0)
    h_iter = co.unassigned_independent_hyperparameter_iterator(output_lst)
    for i, v in enumerate(hyperp_value_lst):
        h = h_iter.next()
        h.assign_value(v)
        draw_fn(i + 1)

    in_filepath_expr = ut.join_paths(
        [out_folderpath, graph_name + '-{0..%d}.pdf' % len(hyperp_value_lst)])
    out_filepath = ut.join_paths([out_folderpath, graph_name + '.pdf'])
    ut.run_bash_command(
        'pdftk %s cat output %s' % (in_filepath_expr, out_filepath))

    for i in xrange(len(hyperp_value_lst) + 1):
        filepath = ut.join_paths([out_folderpath, graph_name + '-%d.pdf' % i])
        ut.delete_file(filepath)


class LinePlot:

    def __init__(self, title=None, xlabel=None, ylabel=None):
        self.data = []
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.color_to_str = {'black': 'k', 'red': 'r'}
        self.line_type_to_str = {'solid': '-', 'dotted': ':', 'dashed': '--'}

    def add_line(self, xs, ys, label=None, err=None, color=None,
                 line_type=None):
        d = {
            "xs": xs,
            "ys": ys,
            "label": label,
            "err": err,
            "color": color,
            "line_type": line_type
        }
        self.data.append(d)

    def plot(self, show=True, fpath=None):
        f = plt.figure()
        for d in self.data:
            fmt = (self.color_to_str.get(d['color'], '') +
                   self.line_type_to_str.get(d['line_type'], ''))
            plt.errorbar(
                d['xs'], d['ys'], yerr=d['err'], label=d['label'], fmt=fmt)

        if self.title is not None:
            plt.title(self.title)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)

        if any([d['label'] is not None for d in self.data]):
            plt.legend(loc='best')

        if fpath is not None:
            f.savefig(fpath, bbox_inches='tight')
        if show:
            plt.show()
        return f
