"""
Search space from Efficient Neural Architecture Search (Pham'17)
"""
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
from collections import OrderedDict

import tensorflow as tf

from dev.architecture_search_benchmarks.helpers import tfeager as htfe
from dev.architecture_search_benchmarks.search_spaces.common_eager import D
from dev.architecture_search_benchmarks.search_spaces.common_ops_eager import (
    conv2D, conv2D_depth_separable, global_pool, dropout, fc_softmax, 
    wrap_batch_norm_relu, avg_pool, max_pool)
import darch.modules as mo

TFEM = htfe.TFEModule

class WeightSharer(object):
    def __init__(self):
        self.name_to_weight = OrderedDict()
        self.weights_used = set()
    
    def get(self, name, weight_fn):
        if name not in self.name_to_weight:
            with tf.device('/gpu:0'):
                self.name_to_weight[name] = weight_fn()
            print(name)
#        self.weights_used.add(name)
#        self.name_to_weight[name].gpu()
        return self.name_to_weight[name]

    def reset(self):
#        for name in self.weights_used:
#            self.name_to_weight[name].cpu()
#        self.weights_used.clear()
        pass
    
# Take in array of boolean hyperparams, concatenate layers corresponding to true
# to form skip connections
def concatenate_skip_layers(h_connects, weight_sharer):
    def cfn(di, dh):
        bn = weight_sharer.get('skip_bn_' + str(len(dh)), tf.keras.layers.BatchNormalization)
        def fn(di, isTraining=True):
            inputs = [di['In' + str(i)] for i in range(len(dh)) if dh['select_' + str(i)]]
            inputs.append(di['In' + str(len(dh))])
            with tf.device('/gpu:0'):
                out = tf.add_n(inputs)
                return {'Out' : bn(out, training=isTraining)}

        return fn
    return TFEM('SkipConcat', 
               {'select_' + str(i) : h_connects[i] for i in range(len(h_connects))},
               cfn, ['In' + str(i) for i in range(len(h_connects) + 1)], ['Out']).get_io()

def enas_conv(out_filters, filter_size, separable, weight_sharer, name):
    io_pair = (conv2D_depth_separable(filter_size, name, weight_sharer) if separable 
               else conv2D(filter_size, name, weight_sharer))
    return mo.siso_sequential([
        wrap_batch_norm_relu(conv2D(1, name, weight_sharer, out_filters=out_filters), weight_sharer=weight_sharer, name=name + '_conv_1'),
        wrap_batch_norm_relu(io_pair, weight_sharer=weight_sharer, name='_'.join([name, str(filter_size), str(separable)]))
    ])

def enas_op(h_op_name, out_filters, name, weight_sharer):
    return mo.siso_or({
        'conv3': lambda: enas_conv(out_filters, 3, False, weight_sharer, name),
        'conv5': lambda: enas_conv(out_filters, 5, False, weight_sharer, name),
        'dsep_conv3': lambda: enas_conv(out_filters, 3, True, weight_sharer, name),
        'dsep_conv5': lambda: enas_conv(out_filters, 5, True, weight_sharer, name),
        'avg_pool': lambda: avg_pool(D([3]), D([1])),
        'max_pool': lambda: max_pool(D([3]), D([1]))
    }, h_op_name)

def enas_repeat_fn(inputs, outputs, layer_id, out_filters, weight_sharer):
    h_enas_op = D(['conv3', 'conv5', 'dsep_conv3', 'dsep_conv5', 'avg_pool', 'max_pool'], name='op_' + str(layer_id))
    #h_enas_op = D(['max_pool'], name='op_' + str(layer_id))
    op_inputs, op_outputs = enas_op(h_enas_op, out_filters, 'op_' + str(layer_id), weight_sharer)
    outputs[list(outputs.keys())[-1]].connect(op_inputs['In'])

    h_connects = [D([True, False], name='skip_'+str(idx)+'_'+str(layer_id)) for idx in range(layer_id - 1)]
    skip_inputs, skip_outputs = concatenate_skip_layers(h_connects, weight_sharer)

    for i in range(len(h_connects)):
        outputs[list(outputs.keys())[i]].connect(skip_inputs['In' + str(i)])
    op_outputs['Out'].connect(skip_inputs['In' + str(len(h_connects))])

    outputs['Out' + str(len(outputs))] = skip_outputs['Out']
    return inputs, outputs
    
def enas_space(h_num_layers, out_filters, fn_first, fn_repeats, input_names, output_names, weight_sharer, scope=None):
    def sub_fn(num_layers):
        assert num_layers > 0
        inputs, outputs = fn_first()
        temp_outputs = OrderedDict(outputs)
        for i in range(1, num_layers + 1):
            inputs, temp_outputs = fn_repeats(inputs, temp_outputs, i, out_filters, weight_sharer)
        return inputs, OrderedDict({'Out': temp_outputs['Out' + str(len(temp_outputs) - 1)]})
    return mo.substitution_module('ENASModule', {'num_layers': h_num_layers}, 
                                  sub_fn, input_names, output_names, scope)

def get_enas_search_space(num_classes, num_layers, out_filters, weight_sharer):
    h_N = D([num_layers], name='num_layers')
    return mo.siso_sequential([
        enas_space(
            h_N, out_filters,
            #mo.empty,
            lambda: wrap_batch_norm_relu(
                conv2D(3, 'stem', weight_sharer, out_filters=out_filters), 
                add_relu=False, weight_sharer=weight_sharer, name='stem'),
            enas_repeat_fn, ['In'], ['Out'], weight_sharer),
        global_pool(),
        dropout(keep_prob=.9),
        fc_softmax(num_classes, weight_sharer),
        ])

class SSFEnasnetEager(mo.SearchSpaceFactory):
    def __init__(self, num_classes, num_layers, out_filters):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes
        self.weight_sharer = WeightSharer()
        self.num_layers = num_layers
        self.out_filters = out_filters

    def _get_search_space(self):
        inputs, outputs = get_enas_search_space(
            self.num_classes,
            self.num_layers, 
         #   1,
            self.out_filters, 
            self.weight_sharer)
        return inputs, outputs, {}
