from __future__ import absolute_import
from builtins import str
from builtins import range
import itertools

import tensorflow as tf

import deep_architect.helpers.tensorflow_eager_support as htfe
from deep_architect.contrib.deep_learning_backend.tf_keras_ops import (
    relu, batch_normalization, conv2d, max_pool2d, fc_layer, dropout, flatten,
    add)

import deep_architect.modules as mo
from deep_architect.hyperparameters import Discrete as D
from deep_architect.hyperparameters import Bool


def genetic_stage(input_fn, node_fn, output_fn, h_connections, num_nodes):

    def substitution_fn(**dh):
        num_ins = [
            sum([dh['%d_%d' % (in_id, out_id)]
                 for in_id in range(1, out_id)])
            for out_id in range(1, num_nodes + 1)
        ]
        num_outs = [
            sum([
                dh['%d_%d' % (in_id, out_id)]
                for out_id in range(in_id + 1, num_nodes + 1)
            ])
            for in_id in range(1, num_nodes + 1)
        ]
        if sum(num_ins) == 0:
            return input_fn()

        nodes = [input_fn()]
        nodes += [
            node_fn(max(num_ins[i], 1))
            if num_ins[i] > 0 or num_outs[i] > 0 else None
            for i in range(num_nodes)
        ]
        nodes.append(
            output_fn(
                len([
                    i for i in range(len(num_outs))
                    if num_outs[i] == 0 and nodes[i + 1] is not None
                ])))

        num_connected = [0] * (num_nodes + 2)
        for in_id in range(1, num_nodes + 1):
            # Connect nodes with no input to original input
            if num_ins[in_id - 1] == 0 and nodes[in_id] is not None:
                nodes[0][1]['Out'].connect(nodes[in_id][0]['In0'])
            # Connect nodes with no output to final output node
            if num_outs[in_id - 1] == 0 and nodes[in_id] is not None:
                nodes[in_id][1]['Out'].connect(nodes[-1][0]['In%d' %
                                                            num_connected[-1]])
                num_connected[-1] += 1

            # Connect internal nodes
            for out_id in range(in_id + 1, num_nodes + 1):
                if dh['%d_%d' % (in_id, out_id)] == 1:
                    nodes[in_id][1]['Out'].connect(
                        nodes[out_id][0]['In' + str(num_connected[out_id])])
                    num_connected[out_id] += 1
        return nodes[0][0], nodes[-1][1]

    i = 0
    name_to_hparam = {}
    for ix, hparam in enumerate(
            itertools.combinations(range(1, num_nodes + 1), 2)):
        name_to_hparam['%d_%d' % hparam] = h_connections[ix]
    return mo.substitution_module('GeneticStage',
                                  name_to_hparam,
                                  substitution_fn, ['In'], ['Out'],
                                  scope=None)


def intermediate_node_fn(num_inputs, filters):
    return mo.siso_sequential([
        add(num_inputs),
        conv2d(D([filters]), D([3])),
        batch_normalization(),
        relu()
    ])


def generate_stage(stage_num, num_nodes, filters, filter_size):
    h_connections = [
        Bool(name='%d_in_%d_%d' % (stage_num, in_id, out_id))
        for (in_id,
             out_id) in itertools.combinations(range(1, num_nodes + 1), 2)
    ]

    return genetic_stage(
        lambda: mo.siso_sequential([
            conv2d(D([filters]), D([filter_size])),
            batch_normalization(),
            relu()
        ]), lambda num_inputs: intermediate_node_fn(num_inputs, filters), lambda
        num_inputs: intermediate_node_fn(num_inputs, filters), h_connections,
        num_nodes)


def generate_search_space(nodes_per_stage, filters_per_stage,
                          filter_size_per_stage):
    search_space = []

    for i in range(len(nodes_per_stage)):
        search_space.append(
            generate_stage(i, nodes_per_stage[i], filters_per_stage[i],
                           filter_size_per_stage[i]))
        search_space.append(max_pool2d(D([3]), D([2]), D(['SAME'])))
    search_space += [
        flatten(),
        fc_layer(D([1024])),
        dropout(D([.5])),
        fc_layer(D([10]))
    ]
    return mo.siso_sequential(search_space)


class SSF_Genetic(mo.SearchSpaceFactory):

    def __init__(self):
        mo.SearchSpaceFactory.__init__(
            self, lambda: generate_search_space([3, 4, 5], [64, 128, 256],
                                                [5, 5, 5]))
