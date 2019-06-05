from __future__ import absolute_import
from builtins import str
from builtins import range
import itertools

import tensorflow as tf
from collections import OrderedDict

import dev.helpers.tfeager as htfe
import deep_architect.modules as mo
import deep_architect.contrib.misc.search_spaces.tensorflow.cnn2d as cnn2d
from dev.deep_learning_backend.tfe_ops import (relu, batch_normalization,
                                               conv2d, separable_conv2d,
                                               avg_pool2d, max_pool2d,
                                               min_pool2d, fc_layer,
                                               global_pool2d, dropout)

from deep_architect.hyperparameters import Discrete as D
from deep_architect.hyperparameters import Bool


def cell(input_fn, node_fn, output_fn, h_connections, num_nodes, channels):

    def dfs(node, visited, dh):
        if visited[node]:
            return False
        if node == num_nodes + 1:
            return True
        visited[node] = True
        return any([
            dfs(i, visited, dh)
            for i in range(node + 1, num_nodes + 2)
            if dh['%d_%d' % (node, i)]
        ])

    def check_path(dh):
        visited = [False] * (num_nodes + 2)
        # visited[0] = True
        return dfs(0, visited, dh)

    def substitution_fn(**dh):
        # if not check_path(dh):
        #     raise ValueError('No path exists between input and output')
        num_ins = [
            sum([dh['%d_%d' % (in_id, out_id)]
                 for in_id in range(out_id)])
            for out_id in range(num_nodes + 2)
        ]
        num_outs = [
            sum([
                dh['%d_%d' % (in_id, out_id)]
                for out_id in range(in_id + 1, num_nodes + 2)
            ])
            for in_id in range(num_nodes + 2)
        ]

        for i in range(1, num_nodes + 1):
            if num_outs[i] > 0 and num_ins[i] == 0:
                raise ValueError('Node exists with no path to input')
            if num_ins[i] > 0 and num_outs[i] == 0:
                raise ValueError('Node exists with no path to output')
        if num_ins[-1] == 0:
            raise ValueError('No path exists between input and output')

        if sum(num_ins) > 9:
            raise ValueError('More than 9 edges')

        if dh['%d_%d' % (0, num_nodes + 1)]:
            num_ins[-1] -= 1
        # if num_ins[-1] == 0:
        #     return input_fn(channels)
        # Compute the number of channels that each vertex outputs
        int_channels = channels // num_ins[-1]
        correction = channels % num_ins[-1]

        vertex_channels = [0] * (num_nodes + 2)
        vertex_channels[-1] = channels
        # Distribute channels as evenly as possible among vertices connected
        # to output vertex
        for in_id in range(1, num_nodes + 1):
            if dh['%d_%d' % (in_id, num_nodes + 1)]:
                vertex_channels[in_id] = int_channels
                if correction > 0:
                    vertex_channels[in_id] += 1
                    correction -= 1
        # For all other vertices, the number of channels they output is the max
        # of the channels outputed by all vertices they flow into
        for in_id in range(num_nodes - 1, 0, -1):
            if not dh['%d_%d' % (in_id, num_nodes + 1)]:
                for out_id in range(in_id + 1, num_nodes + 1):
                    if dh['%d_%d' % (in_id, out_id)]:
                        vertex_channels[in_id] = max(vertex_channels[in_id],
                                                     vertex_channels[out_id])

        nodes = [mo.identity()]
        nodes += [
            node_fn(num_ins[i], i -
                    1, vertex_channels[i]) if num_outs[i] > 0 else None
            for i in range(1, num_nodes + 1)
        ]
        nodes.append(output_fn(num_ins[num_nodes + 1]))
        num_connected = [0] * (num_nodes + 2)
        # Project input vertex to correct dimensions if used
        for out_id in range(1, num_nodes + 1):
            if dh['%d_%d' % (0, out_id)]:
                proj_in, proj_out = input_fn(vertex_channels[out_id])
                nodes[0][1]['Out'].connect(proj_in['In'])
                proj_out['Out'].connect(
                    nodes[out_id][0]['In' + str(num_connected[out_id])])
                num_connected[out_id] += 1
        for in_id in range(1, num_nodes + 1):
            for out_id in range(in_id + 1, num_nodes + 2):
                if dh['%d_%d' % (in_id, out_id)]:
                    nodes[in_id][1]['Out'].connect(
                        nodes[out_id][0]['In' + str(num_connected[out_id])])
                    num_connected[out_id] += 1
        if dh['%d_%d' % (0, num_nodes + 1)]:
            proj_in, proj_out = input_fn(channels)
            add_in, add_out = add(2)
            nodes[0][1]['Out'].connect(proj_in['In'])
            proj_out['Out'].connect(add_in['In0'])
            nodes[-1][1]['Out'].connect(add_in['In1'])
            nodes[-1] = (add_in, add_out)

        return nodes[0][0], nodes[-1][1]

    i = 0
    name_to_hparam = {}
    for ix, hparam in enumerate(itertools.combinations(range(num_nodes + 2),
                                                       2)):
        name_to_hparam['%d_%d' % hparam] = h_connections[ix]
    return mo.substitution_module('NasbenchCell',
                                  name_to_hparam,
                                  substitution_fn, ['In'], ['Out'],
                                  scope=None)


def add(num_inputs):

    def compile_fn(di, dh):
        in_channels = [tf.shape(di[inp])[-1] for inp in di]
        final_channels = tf.reduce_min(in_channels)

        def forward_fn(di, isTraining=True):
            out = [di[inp][:, :, :, :final_channels] for inp in di]
            return {'Out': tf.add_n(out)}

        return forward_fn

    return htfe.TFEModule('Add', {}, compile_fn,
                          ['In' + str(i) for i in range(num_inputs)],
                          ['Out']).get_io()


def intermediate_node_fn(num_inputs, node_id, filters, cell_ops):
    return mo.siso_sequential([
        add(num_inputs),
        mo.siso_or(
            {
                'conv1': lambda: conv2d(D([filters]), D([1])),
                'conv3': lambda: conv2d(D([filters]), D([3])),
                'max3': lambda: max_pool2d(D([3]))
            }, cell_ops[node_id]),
        batch_normalization(),
        relu()
    ])


def concat(num_ins):

    def compile_fn(di, dh):

        def forward_fn(di, isTraining=True):
            return {
                'Out':
                tf.concat(values=[di[input_name] for input_name in di], axis=3)
            }

        return forward_fn

    return htfe.TFEModule('Concat', {}, compile_fn,
                          ['In' + str(i) for i in range(num_ins)],
                          ['Out']).get_io()


def create_cell_generator(num_nodes):
    h_connections = [
        Bool(name='in_%d_%d' % (in_id, out_id))
        for (in_id, out_id) in itertools.combinations(range(num_nodes + 2), 2)
    ]

    cell_ops = [
        D(['conv1', 'conv3', 'max3'], name='node_%d' % i)
        for i in range(num_nodes)
    ]

    def generate(filters):
        return cell(
            lambda channels: mo.siso_sequential(
                [conv2d(D([channels]), D([1])),
                 batch_normalization(),
                 relu()]), lambda num_inputs, node_id, channels:
            intermediate_node_fn(num_inputs, node_id, channels, cell_ops),
            concat, h_connections, 5, filters)

    return generate


def stem():
    return mo.siso_sequential([
        conv2d(D([128]), D([3])),
        batch_normalization(),
        relu(),
    ])


def generate_search_space(stacks, num_cells_per_stack, num_nodes_per_cell,
                          num_init_filters):
    search_space = [stem()]
    cell_fn = create_cell_generator(num_nodes_per_cell)
    num_filters = num_init_filters
    for i in range(stacks):
        if i > 0:
            search_space.append(max_pool2d(D([2]), D([2])))
            num_filters *= 2
        for j in range(num_cells_per_stack):
            search_space.append(cell_fn(num_filters))
    search_space += [global_pool2d(), fc_layer(D([10]))]
    return mo.siso_sequential(search_space)


class SSF_Nasbench(mo.SearchSpaceFactory):

    def __init__(self):
        mo.SearchSpaceFactory.__init__(
            self, lambda: generate_search_space(3, 3, 5, 128))
