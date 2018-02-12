
import torch
import torch.nn as nn
from graphviz import Digraph
from torch.autograd import Variable

# NOTE: no information about the other models now. the problem is how it is 
# done. maybe could have tags.
# TODO: probably it makes sense to have a notion of the batch dimension. 
# there is always going to exist the time dimension.
class EmbeddingEncoder(nn.Module):
    def __init__(self, emb_nums, emb_dims, freeze_embs, 
        num_bos_pad=0, num_eos_pad=0):
        assert len(emb_nums) == len(emb_dims)

        self.embs = [nn.Embedding(n, d) for (n, d) in zip(emb_nums, emb_dims)]

        self.all_embs = []
        self.all_embs.extend(self.embs)

        self.num_bos_pad = num_bos_pad
        if num_bos_pad > 0:
            self.bos_embs = [nn.Embedding(1, d) for d in emb_dims]
            self.all_embs.extend(self.bos_embs)

        self.num_eos_pad = num_eos_pad
        if num_eos_pad > 0:
            self.eos_embs = [nn.Embedding(1, d) for d in emb_dims]
            self.all_embs.extend(self.eos_embs)

        if freeze_embs:
            for e in self.all_embs:
                e.weight.requires_grad = False

    # careful about the concatenation specification.
    # TODO: 
    def encode(self, xs):

        for i in xrange( len(self.embs) ):
            pass
        
    ### do the paddings. do some assumption on

def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot
