from __future__ import absolute_import
from builtins import str
from builtins import range

import tensorflow as tf
import numpy as np

import deep_architect.core as co
import deep_architect.hyperparameters as hp
import deep_architect.helpers.tensorflow as htf
import deep_architect.modules as mo
import deep_architect.contrib.misc.search_spaces.tensorflow.cnn2d as cnn2d
from deep_architect.contrib.deep_learning_backend.tf_ops import (
    relu, batch_normalization, conv2d, conv2d_separable, add, avg_pool2d,
    max_pool2d, fc_layer, global_pool2d
)

from deep_architect.hyperparameters import Discrete as D


TFM = htf.TensorflowModule
const_fn = lambda c: lambda shape: tf.constant(c, shape=shape)

def conv_spatial_separable(h_num_filters, h_filter_size, h_stride):
    h_filter_size_1 = co.DependentHyperparameter(lambda size: [1, size], {'size': h_filter_size})
    h_filter_size_2 = co.DependentHyperparameter(lambda size: [size, 1], {'size': h_filter_size})
    return mo.siso_sequential([
        conv2d(h_num_filters, h_filter_size_1, h_stride, D([1]), D([True])),
        batch_normalization(),
        relu(),
        conv2d(h_num_filters, h_filter_size_2, h_stride, D([1]), D([True]))
    ])

def wrap_relu_batch_norm(conv):
    return mo.siso_sequential([
        relu(),
        conv,
        batch_normalization()
    ])

def apply_conv_op(main_op, op_name, h_num_filter):
    if op_name is 's_sep3':
        reduced_filter_size = co.DependentHyperparameter(
            lambda size: int(3 * size / 8), 
            {'size': h_num_filter})
    else:
        reduced_filter_size = co.DependentHyperparameter(
            lambda size: int(size / 4), 
            {'size': h_num_filter})

    bottleneck_too_thin = co.DependentHyperparameter(
        lambda size: size < 1, {'size': reduced_filter_size}
    )

    return mo.siso_sequential([
        mo.siso_optional(
            lambda: wrap_relu_batch_norm(
                conv2d(reduced_filter_size, D([1]), D([1]), D([1]), D([True]))
            ), bottleneck_too_thin
        ),
        wrap_relu_batch_norm(main_op),
        mo.siso_optional(
            lambda: wrap_relu_batch_norm(
                conv2d(h_num_filter, D([1]), D([1]), D([1]), D([True])),
            ), bottleneck_too_thin
        )
    ])

def check_stride(h_stride, h_num_filter, h_op_name):
    need_stride = co.DependentHyperparameter(
        lambda name, stride: int(name is 'conv1' or stride > 1),
        {'name': h_op_name, 'stride': h_stride}
    )
    return mo.siso_or([
        mo.empty,
        lambda: wrap_relu_batch_norm(
            conv2d(h_num_filter, D([1]), h_stride, D([1]), D([True]))
        )
    ], need_stride)

def check_reduce(main_op, h_stride, h_num_filter):
    need_stride = co.DependentHyperparameter(
        lambda stride: int(stride > 1),
        {'stride': h_stride}
    )
    return mo.siso_sequential([
        main_op,
        mo.siso_optional(
            lambda: mo.siso_sequential([
                conv2d(h_num_filter, D([1]), h_stride, D([1]), D([True])),
                batch_normalization()
            ]), need_stride
        )
    ])

def stacked_depth_separable_conv(h_filter_size, h_num_filters, 
                                 h_depth_multiplier, h_stride):
    return mo.siso_sequential([
        relu(),
        conv2d_separable(
            h_num_filters, h_filter_size, D([1]), D([1]), h_depth_multiplier, 
            D([True])),
        batch_normalization(),
        relu(),
        conv2d_separable(
            h_num_filters, h_filter_size, h_stride, D([1]), h_depth_multiplier, 
            D([True])),
        batch_normalization()
    ])


# The operations used in search space 1 and search space 3 in Regularized Evolution for
# Image Classifier Architecture Search (Real et al, 2018)
def sp1_operation(h_op_name, h_stride, h_filters):
    return mo.siso_or({
            'identity': lambda: check_stride(h_stride, h_filters, h_op_name),
            'd_sep3': lambda: stacked_depth_separable_conv(D([3]), h_filters, D([1]), h_stride),
            'd_sep5': lambda: stacked_depth_separable_conv(D([5]), h_filters, D([1]), h_stride),
            'd_sep7': lambda: stacked_depth_separable_conv(D([7]), h_filters, D([1]), h_stride),
            'avg3': lambda: check_reduce(avg_pool2d(D([3]), h_stride), h_stride, h_filters),
            'max3': lambda: check_reduce(max_pool2d(D([3]), h_stride), h_stride, h_filters),
            'dil3': lambda: apply_conv_op(
                conv2d(h_filters, D([3]), h_stride, D([2]), D([True])),
                'dil3', h_stride),
            's_sep7': lambda:apply_conv_op(
                conv_spatial_separable(h_filters, D([7]), h_stride),
                's_sep7', h_stride), 
            }, h_op_name)

# The operations used in search space 2 in Regularized Evolution for
# Image Classifier Architecture Search (Real et al, 2018)
def sp2_operation(h_op_name, h_stride, h_filters):
    return mo.siso_or({
            'identity': lambda: mo.empty(),
            'conv1': lambda: check_stride(h_stride, h_filters, h_op_name),
            'conv3': lambda: apply_conv_op(
                conv2d(h_filters, D([3]), h_stride, D([1]), D([True])),
                'conv3', h_stride),
            'd_sep3': lambda: stacked_depth_separable_conv(D([3]), h_filters, D([1]), h_stride),
            'd_sep5': lambda: stacked_depth_separable_conv(D([5]), h_filters, D([1]), h_stride),
            'd_sep7': lambda: stacked_depth_separable_conv(D([7]), h_filters, D([1]), h_stride),
            'avg2': lambda: check_reduce(avg_pool2d(D([2]), h_stride), h_stride, h_filters),
            'avg3': lambda: check_reduce(avg_pool2d(D([3]), h_stride), h_stride, h_filters),
            'max2': lambda: check_reduce(max_pool2d(D([2]), h_stride), h_stride, h_filters),
            'max3': lambda: check_reduce(max_pool2d(D([3]), h_stride), h_stride, h_filters),
            'dil3_2': lambda: apply_conv_op(
                conv2d(h_filters, D([3]), h_stride, D([2]), D([True])), 
                'dil3_2', h_stride),
            'dil4_2': lambda: apply_conv_op(
                conv2d(h_filters, D([3]), h_stride, D([4]), D([True])), 
                'dil3_4', h_stride),
            'dil6_2': lambda: apply_conv_op(
                conv2d(h_filters, D([3]), h_stride, D([6]), D([True])), 
                'dil3_6', h_stride),
            's_sep3': lambda:apply_conv_op(
                conv_spatial_separable(h_filters, D([3]), h_stride),
                's_sep3', h_stride),
            's_sep7': lambda:apply_conv_op(
                conv_spatial_separable(h_filters, D([7]), h_stride),
                's_sep7', h_stride),             
            }, h_op_name)

# A module that takes in a specifiable number of inputs, uses 1x1 convolutions to make the number
# filters match, and then adds the inputs together
def mi_add(num_terms):
    def cfn(di, dh):
        (_, _, _, min_channels) = di['In0'].get_shape().as_list()
        for i in range(num_terms):
            (_, _, _, channels) = di['In' + str(i)].get_shape().as_list()
            min_channels = channels if channels < min_channels else min_channels

        W_init_fn = cnn2d.kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)
        w_dict = {}
        b_dict = {}
        for i in range(num_terms):
            (_, _, _, channels) = di['In' + str(i)].get_shape().as_list()
            if channels != min_channels:
                w_dict[i] = tf.Variable( W_init_fn( [1, 1, channels, min_channels]))
                b_dict[i] = tf.Variable( b_init_fn( [min_channels]))

        def fn(In):
            trans = []
            for i in range(num_terms):
                if i in w_dict:
                    trans.append(tf.nn.bias_add(tf.nn.conv2d(di['In' + str(i)], w_dict[i], [1, 1, 1, 1], 'SAME'), b_dict[i]))
                else:
                    trans.append(di['In' + str(i)])
            total_sum = trans[0]
            for i in range(1, num_terms):
                total_sum = tf.add(total_sum, trans[i])
            return {'Out' : total_sum}
        return fn
    return TFM('MultiInputAdd', {}, cfn, ['In' + str(i) for i in range(num_terms)], ['Out']).get_io()

# A module that applies a sp1_operation to two inputs, and adds them together
def sp1_combine(h_op1_name, h_op2_name, h_stride, h_filters):
    in1 = lambda: sp1_operation(h_op1_name, h_stride, h_filters)
    in2 = lambda: sp1_operation(h_op2_name, h_stride, h_filters)
    return mo.mimo_combine([in1, in2], mi_add)

# A module that selects an input from a list of inputs
def selector(h_selection, sel_fn, num_selections):
    def cfn(di, dh):
        selection = dh['selection']
        sel_fn(selection)
        def fn(di):
            return {'Out': di['In' + str(selection)]}
        return fn
    return TFM('Selector', {'selection': h_selection}, cfn, ['In' + str(i) for i in range(num_selections)], ['Out']).get_io()

# A module that outlines the basic cell structure in Learning Transferable
# Architectures for Scalable Image Recognition (Zoph et al, 2017)
def basic_cell(h_sharer, h_filters, C=5, normal=True):
    i_inputs, i_outputs = mo.empty(num_connections=2)
    available = [i_outputs['Out' + str(i)] for i in range(len(i_outputs))]
    unused = [False] * (C + 2)
    for i in range(C):
        if normal:
            h_op1 = h_sharer.get('h_norm_op1_' + str(i))
            h_op2 = h_sharer.get('h_norm_op2_' + str(i))
            h_in0_pos = h_sharer.get('h_norm_in0_pos_' + str(i))
            h_in1_pos = h_sharer.get('h_norm_in1_pos_' + str(i))
        else:
            h_op1 = h_sharer.get('h_red_op1_' + str(i))
            h_op2 = h_sharer.get('h_red_op2_' + str(i))
            h_in0_pos = h_sharer.get('h_red_in0_pos_' + str(i))
            h_in1_pos = h_sharer.get('h_red_in1_pos_' + str(i))

        def select(selection):
            unused[selection] = True

        sel0_inputs, sel0_outputs = selector(h_in0_pos, select, len(available))
        sel1_inputs, sel1_outputs = selector(h_in1_pos, select, len(available))

        for o_idx in range(len(available)):
            available[o_idx].connect(sel0_inputs['In' + str(o_idx)])
            available[o_idx].connect(sel1_inputs['In' + str(o_idx)])

        comb_inputs, comb_outputs = sp1_combine(
            h_op1, h_op2, D([1 if normal or i > 0 else 2]), h_filters)
        sel0_outputs['Out'].connect(comb_inputs['In0'])
        sel1_outputs['Out'].connect(comb_inputs['In1'])
        available.append(comb_outputs['Out'])

    unused_outs = [available[i] for i in range(len(unused)) if not unused[i]]
    s_inputs, s_outputs = mi_add(len(unused_outs))
    for i in range(len(unused_outs)):
        unused_outs[i].connect(s_inputs['In' + str(i)])
    return i_inputs, s_outputs

# A module that creates a number of repeated cells (both reduction and normal)
# as in Learning Transferable Architectures for Scalable Image Recognition (Zoph et al, 2017)
def ss_repeat(h_N, h_sharer, h_filters, C, num_ov_repeat, num_classes, scope=None):
    def sub_fn(N):
        assert N > 0

        i_inputs, i_outputs = mo.simo_split(2)
        prev_2 = [i_outputs['Out0'], i_outputs['Out1']]
        h_iter = h_filters
        for i in range(num_ov_repeat):
            for j in range(N):
                print(h_filters)
                norm_inputs, norm_outputs = basic_cell(h_sharer, h_iter, C=C, normal=True)
                prev_2[0].connect(norm_inputs['In0'])
                prev_2[1].connect(norm_inputs['In1'])
                prev_2[0] = prev_2[1]
                prev_2[1] = norm_outputs['Out']
            if j < num_ov_repeat - 1:
                h_iter = co.DependentHyperparameter(
                    lambda filters: filters*2,
                    {'filters': h_iter})

                red_inputs, red_outputs = basic_cell(h_sharer, h_iter, C=C, normal=False)
                prev_2[0].connect(red_inputs['In0'])
                prev_2[1].connect(red_inputs['In1'])
                prev_2[0] = prev_2[1]
                prev_2[1] = red_outputs['Out']

        logits_inputs, logits_outputs = mo.siso_sequential([
            global_pool2d(),
            fc_layer(D([num_classes]))
        ]) 
        prev_2[1].connect(logits_inputs['In'])
        return i_inputs, logits_outputs
    return mo.substitution_module('SS_repeat', {'N': h_N}, sub_fn, ['In'], ['Out'], scope)

# Creates search space using operations from search spaces 1 and 3 from
# Regularized Evolution for Image Classifier Architecture Search (Real et al, 2018)
def get_search_space_small(num_classes, C):
    co.Scope.reset_default_scope()
    C = 5
    h_N = D([2])
    h_F = D([24])
    h_sharer = hp.HyperparameterSharer()
    for i in range(C):
        h_sharer.register('h_norm_op1_' + str(i), lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 'dil3', 's_sep7'], name='Mutatable'))
        h_sharer.register('h_norm_op2_' + str(i), lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 'dil3', 's_sep7'], name='Mutatable'))
        h_sharer.register('h_norm_in0_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
        h_sharer.register('h_norm_in1_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
    for i in range(C):
        if i == 0:
            h_sharer.register('h_red_op1_' + str(i), lambda: D(['d_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 's_sep7'], name='Mutatable'))
            h_sharer.register('h_red_op2_' + str(i), lambda: D(['d_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 's_sep7'], name='Mutatable'))
        else:
            h_sharer.register('h_red_op1_' + str(i), lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 'dil3', 's_sep7'], name='Mutatable'))
            h_sharer.register('h_red_op2_' + str(i), lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 'dil3', 's_sep7'], name='Mutatable'))
        h_sharer.register('h_red_in0_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
        h_sharer.register('h_red_in1_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
    i_inputs, i_outputs = conv2d(h_F, D([1]), D([1]), D([1]), D([True]))
    o_inputs, o_outputs = ss_repeat(h_N, h_sharer, h_F, C, 3, num_classes)
    i_outputs['Out'].connect(o_inputs['In'])
    r_inputs, r_outputs = mo.siso_sequential([mo.empty(), (i_inputs, o_outputs), mo.empty()])
    return r_inputs, r_outputs

# Search space 1 from Regularized Evolution for 
# Image Classifier Architecture Search (Real et al, 2018)
def get_search_space_1(num_classes):
    return get_search_space_small(num_classes, 5)

# Search space 3 from Regularized Evolution for Image Classifier Architecture Search (Real et al, 2018)
def get_search_space_3(num_classes):
    return get_search_space_small(num_classes, 15)

# Search space 2 from Regularized Evolution for Image Classifier Architecture Search (Real et al, 2018)
def get_search_space_2(num_classes):
    co.Scope.reset_default_scope()
    C = 5
    h_N = D([3])
    h_F = D([8])
    h_sharer = hp.HyperparameterSharer()
    for i in range(C):
        h_sharer.register('h_norm_op1_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
          'min2', 'max2', 'manorm_x3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6'], name='Mutatable'))
        h_sharer.register('h_norm_op2_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
          'min2', 'max2', 'manorm_x3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6'], name='Mutatable'))
        h_sharer.register('h_in0_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
        h_sharer.register('h_in1_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
    for i in range(C):
        if i == 0:
            h_sharer.register('h_red_op1_' + str(i), lambda: D(['conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
            'min2', 'max2', 'max3', 's_sep3' 's_sep7'], name='Mutatable'))
            h_sharer.register('h_red_op2_' + str(i), lambda: D(['conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
            'min2', 'max2', 'max3', 's_sep3' 's_sep7'], name='Mutatable'))
        else:
            h_sharer.register('h_red_op1_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
            'min2', 'max2', 'max3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6'], name='Mutatable'))
            h_sharer.register('h_red_op2_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
            'min2', 'max2', 'max3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6'], name='Mutatable'))
        h_sharer.register('h_red_in0_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
        h_sharer.register('h_red_in1_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
    i_inputs, i_outputs = conv1x1(h_F)
    o_inputs, o_outputs = ss_repeat(h_N, h_sharer, C, 3, num_classes)
    i_outputs['Out'].connect(o_inputs['In'])
    return i_inputs, o_outputs

class SSFZoph17(mo.SearchSpaceFactory):
    def __init__(self, search_space, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

        if search_space == 'sp1':
            self.search_space_fn = get_search_space_1
        elif search_space == 'sp2':
            self.search_space_fn = get_search_space_2
        elif search_space == 'sp3':
            self.search_space_fn = get_search_space_3

    def _get_search_space(self):
        inputs, outputs = self.search_space_fn(self.num_classes)
        return inputs, outputs, {}
