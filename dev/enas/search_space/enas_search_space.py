"""
Search space from Efficient Neural Architecture Search (Pham'17)
"""
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
from collections import OrderedDict

import tensorflow as tf
import numpy as np

from deep_architect.helpers import tensorflow_eager_support as htfe
from deep_architect.hyperparameters import D
from dev.enas.search_space.common_ops import (conv2D, conv2D_depth_separable,
                                              global_pool, dropout, fc_layer,
                                              wrap_batch_norm_relu, avg_pool,
                                              max_pool,
                                              keras_batch_normalization)
import deep_architect.modules as mo

TFEM = htfe.TensorflowEagerModule


class WeightSharer(object):

    def __init__(self, isSharing):
        self.name_to_weight = {}
        self.name_to_np_fn = {}
        self.weight_dict = {}
        self.isSharing = isSharing

    def get(self, name, construct_fn, np_fn):
        if self.isSharing:
            if name not in self.name_to_weight:
                with tf.device('/gpu:0'):
                    self.name_to_weight[name] = construct_fn()
                    self.name_to_np_fn[name] = np_fn
                print(name)

    #        self.weights_used.add(name)
    #        self.name_to_weight[name].gpu()
            return self.name_to_weight[name]
        return construct_fn()

    def load_weights(self, name):
        if name in self.weight_dict:
            return self.weight_dict[name]
        else:
            return None

    def save(self, filename):
        weight_dict = self.weight_dict
        for name in self.name_to_weight:
            weight_dict[name] = self.name_to_np_fn[name](
                self.name_to_weight[name])
        np.save(filename, weight_dict)

    def load(self, filename):
        self.weight_dict = np.load(filename).item()


# Take in array of boolean hyperparams, concatenate layers corresponding to true
# to form skip connections
def concatenate_skip_layers(h_connects, weight_sharer):

    def compile_fn(di, dh):

        def fn(di, is_training=True):
            inputs = [
                di['In' + str(i)]
                for i in range(len(dh))
                if dh['select_' + str(i)]
            ]
            inputs.append(di['In' + str(len(dh))])
            with tf.device('/gpu:0'):
                out = tf.add_n(inputs)
                return {'Out': tf.add_n(inputs)}

        return fn

    return TFEM(
        'SkipConcat',
        {'select_' + str(i): h_connects[i] for i in range(len(h_connects))},
        compile_fn, ['In' + str(i) for i in range(len(h_connects) + 1)],
        ['Out']).get_io()


def enas_conv(out_filters, filter_size, separable, weight_sharer, name):
    io_pair = (conv2D_depth_separable(filter_size, name, weight_sharer)
               if separable else conv2D(filter_size, name, weight_sharer))
    return mo.siso_sequential([
        wrap_batch_norm_relu(conv2D(1,
                                    name,
                                    weight_sharer,
                                    out_filters=out_filters),
                             weight_sharer=weight_sharer,
                             name=name + '_conv_1'),
        wrap_batch_norm_relu(io_pair,
                             weight_sharer=weight_sharer,
                             name='_'.join(
                                 [name, str(filter_size),
                                  str(separable)]))
    ])


def enas_op(h_op_name, out_filters, name, weight_sharer):
    return mo.siso_or(
        {
            'conv3':
            lambda: enas_conv(out_filters, 3, False, weight_sharer, name),
            'conv5':
            lambda: enas_conv(out_filters, 5, False, weight_sharer, name),
            'dsep_conv3':
            lambda: enas_conv(out_filters, 3, True, weight_sharer, name),
            'dsep_conv5':
            lambda: enas_conv(out_filters, 5, True, weight_sharer, name),
            'avg_pool':
            lambda: avg_pool(D([3]), D([1])),
            'max_pool':
            lambda: max_pool(D([3]), D([1]))
        }, h_op_name)


def enas_repeat_fn(inputs, outputs, layer_id, out_filters, weight_sharer):
    h_enas_op = D(
        ['conv3', 'conv5', 'dsep_conv3', 'dsep_conv5', 'avg_pool', 'max_pool'],
        name='op_' + str(layer_id))
    #h_enas_op = D(['max_pool'], name='op_' + str(layer_id))
    op_inputs, op_outputs = enas_op(h_enas_op, out_filters,
                                    'op_' + str(layer_id), weight_sharer)
    outputs[list(outputs.keys())[-1]].connect(op_inputs['In'])

    #Skip connections
    h_connects = [
        D([True, False], name='skip_' + str(idx) + '_' + str(layer_id))
        for idx in range(layer_id - 1)
    ]
    skip_inputs, skip_outputs = concatenate_skip_layers(h_connects,
                                                        weight_sharer)
    for i in range(len(h_connects)):
        outputs[list(outputs.keys())[i]].connect(skip_inputs['In' + str(i)])
    op_outputs['Out'].connect(skip_inputs['In' + str(len(h_connects))])

    # Batch norm after skip
    bn_inputs, bn_outputs = keras_batch_normalization(
        name='skip_bn_' + str(len(h_connects)), weight_sharer=weight_sharer)
    skip_outputs['Out'].connect(bn_inputs['In'])
    outputs['Out' + str(len(outputs))] = bn_outputs['Out']
    return inputs, outputs


def enas_space(h_num_layers,
               out_filters,
               fn_first,
               fn_repeats,
               input_names,
               output_names,
               weight_sharer,
               scope=None):

    def substitution_fn(num_layers):
        assert num_layers > 0
        inputs, outputs = fn_first()
        temp_outputs = OrderedDict(outputs)
        for i in range(1, num_layers + 1):
            inputs, temp_outputs = fn_repeats(inputs, temp_outputs, i,
                                              out_filters, weight_sharer)
        return inputs, OrderedDict(
            {'Out': temp_outputs['Out' + str(len(temp_outputs) - 1)]})

    return mo.substitution_module('ENASModule', {'num_layers': h_num_layers},
                                  substitution_fn, input_names, output_names,
                                  scope)


def get_enas_search_space(num_classes, num_layers, out_filters, weight_sharer):
    h_N = D([num_layers], name='num_layers')
    return mo.siso_sequential([
        enas_space(
            h_N,
            out_filters,
            #mo.empty,
            lambda: wrap_batch_norm_relu(conv2D(
                3, 'stem', weight_sharer, out_filters=out_filters),
                                         add_relu=False,
                                         weight_sharer=weight_sharer,
                                         name='stem'),
            enas_repeat_fn,
            ['In'],
            ['Out'],
            weight_sharer),
        global_pool(),
        dropout(keep_prob=.9),
        fc_layer(num_classes, 'softmax', weight_sharer),
    ])


class SSFEnasnet(mo.SearchSpaceFactory):

    def __init__(self, num_classes, num_layers, out_filters, isSharing=True):
        mo.SearchSpaceFactory.__init__(self, self._get_search_space)
        self.num_classes = num_classes
        self.weight_sharer = WeightSharer(isSharing)
        self.num_layers = num_layers
        self.out_filters = out_filters

    def _get_search_space(self):
        inputs, outputs = get_enas_search_space(self.num_classes,
                                                self.num_layers,
                                                self.out_filters,
                                                self.weight_sharer)
        return inputs, outputs, {}
