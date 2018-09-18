from __future__ import absolute_import
from builtins import str
from builtins import range

import tensorflow as tf
import numpy as np
from pprint import pprint
from collections import OrderedDict

import deep_architect.core as co
import deep_architect.hyperparameters as hp
import deep_architect.helpers.tensorflow as htf
import deep_architect.modules as mo
import deep_architect.contrib.misc.search_spaces.tensorflow.cnn2d as cnn2d
from deep_architect.contrib.deep_learning_backend.tf_ops import (
    relu, batch_normalization, conv2d, separable_conv2d, add, avg_pool2d,
    max_pool2d, fc_layer, global_pool2d
)

from deep_architect.hyperparameters import Discrete as D


TFM = htf.TensorflowModule
const_fn = lambda c: lambda shape: tf.constant(c, shape=shape)

def conv_spatial_separable(h_num_filters, h_filter_size, h_stride):
    h_filter_size_1 = co.DependentHyperparameter(lambda size: [1, size], {'size': h_filter_size})
    h_filter_size_2 = co.DependentHyperparameter(lambda size: [size, 1], {'size': h_filter_size})
    h_stride_1 = co.DependentHyperparameter(lambda stride: [1, stride], {'stride': h_stride})
    h_stride_2 = co.DependentHyperparameter(lambda stride: [stride, 1], {'stride': h_stride})
    return mo.siso_sequential([
        conv2d(h_num_filters, h_filter_size_1, h_stride_1, D([1]), D([True])),
        batch_normalization(),
        relu(),
        conv2d(h_num_filters, h_filter_size_2, h_stride_2, D([1]), D([True]))
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
            {'size': h_num_filter}, name='reduced_filter_size')
    else:
        reduced_filter_size = co.DependentHyperparameter(
            lambda size: int(size / 4), 
            {'size': h_num_filter}, name='reduced_filter_size')

    bottleneck_too_thin = co.DependentHyperparameter(
        lambda size: size < 1, {'size': reduced_filter_size}, name='bottleneck_too_thin'
    )

    bottleneck_filters = co.DependentHyperparameter(
        lambda reduced_filter_size, num_filter: num_filter if reduced_filter_size < 1 else reduced_filter_size, 
        {
            'reduced_filter_size': reduced_filter_size,
            'num_filter': h_num_filter
        }, name='bottleneck_filters'
    )

    return mo.siso_sequential([
        mo.siso_optional(
            lambda: wrap_relu_batch_norm(
                conv2d(reduced_filter_size, D([1]), D([1]), D([1]), D([True]))
            ), bottleneck_too_thin
        ),
        wrap_relu_batch_norm(main_op(bottleneck_filters)),
        mo.siso_optional(
            lambda: wrap_relu_batch_norm(
                conv2d(h_num_filter, D([1]), D([1]), D([1]), D([True])),
            ), bottleneck_too_thin
        )
    ])

def check_stride(h_stride, h_num_filter, h_op_name):
    need_stride = co.DependentHyperparameter(
        lambda name, stride: int(name is 'conv1' or stride > 1),
        {'name': h_op_name, 'stride': h_stride}, name='check_stride'
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
        {'stride': h_stride}, name='check_reduce'
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
        separable_conv2d(
            h_num_filters, h_filter_size, D([1]), D([1]), h_depth_multiplier, 
            D([True])),
        batch_normalization(),
        relu(),
        separable_conv2d(
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
            'avg3': lambda: check_reduce(avg_pool2d(D([3]), D([1])), h_stride, h_filters),
            'max3': lambda: check_reduce(max_pool2d(D([3]), D([1])), h_stride, h_filters),
            'dil3_2': lambda: apply_conv_op(
                lambda filters: conv2d(filters, D([3]), h_stride, D([2]), D([True])),
                'dil3_2', h_filters),
            's_sep7': lambda:apply_conv_op(
                lambda filters: conv_spatial_separable(filters, D([7]), h_stride),
                's_sep7', h_filters), 
            }, h_op_name)

# The operations used in search space 2 in Regularized Evolution for
# Image Classifier Architecture Search (Real et al, 2018)
def sp2_operation(h_op_name, h_stride, h_filters):
    return mo.siso_or({
            'identity': lambda: mo.empty(),
            'conv1': lambda: check_stride(h_stride, h_filters, h_op_name),
            'conv3': lambda: apply_conv_op(
                lambda filters: conv2d(filters, D([3]), h_stride, D([1]), D([True])),
                'conv3', h_stride),
            'd_sep3': lambda: stacked_depth_separable_conv(D([3]), h_filters, D([1]), h_stride),
            'd_sep5': lambda: stacked_depth_separable_conv(D([5]), h_filters, D([1]), h_stride),
            'd_sep7': lambda: stacked_depth_separable_conv(D([7]), h_filters, D([1]), h_stride),
            'avg2': lambda: check_reduce(avg_pool2d(D([2]), D([1])), h_stride, h_filters),
            'avg3': lambda: check_reduce(avg_pool2d(D([3]), D([1])), h_stride, h_filters),
            'max2': lambda: check_reduce(max_pool2d(D([2]), D([1])), h_stride, h_filters),
            'max3': lambda: check_reduce(max_pool2d(D([3]), D([1])), h_stride, h_filters),
            'dil3_2': lambda: apply_conv_op(
                lambda filters: conv2d(filters, D([3]), h_stride, D([2]), D([True])), 
                'dil3_2', h_stride),
            'dil4_2': lambda: apply_conv_op(
                conv2d(h_filters, D([3]), h_stride, D([4]), D([True])), 
                'dil3_4', h_stride),
            'dil6_2': lambda: apply_conv_op(
                lambda filters: conv2d(filters, D([3]), h_stride, D([6]), D([True])), 
                'dil3_6', h_stride),
            's_sep3': lambda:apply_conv_op(
                lambda filters: conv_spatial_separable(filters, D([3]), h_stride),
                's_sep3', h_stride),
            's_sep7': lambda:apply_conv_op(
                lambda filters: conv_spatial_separable(filters, D([7]), h_stride),
                's_sep7', h_stride),             
            }, h_op_name)

def mimo_combine(fns, combine_fn, scope=None, name=None):
    inputs_lst, outputs_lst = zip(*[fns[i]() for i in range(len(fns))])
    c_inputs, c_outputs = combine_fn(len(fns))
    i_inputs = OrderedDict()
    for i in range(len(fns)):
        i_inputs['In' + str(i)] = inputs_lst[i]['In']
        c_inputs['In' + str(i)].connect( outputs_lst[i]['Out'] )
    return (i_inputs, c_outputs)

# A module that takes in a specifiable number of inputs, uses 1x1 convolutions to make the number
# filters match, and then adds the inputs together
def mi_add(num_terms, name=None):
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

        def fn(di):
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
    return TFM('MultiInputAdd' if name is None else name, {}, cfn, ['In' + str(i) for i in range(num_terms)], ['Out']).get_io()

# A module that applies a sp1_operation to two inputs, and adds them together
def sp1_combine(h_op1_name, h_op2_name, h_stride_1, h_stride_2, h_filters):
    op1 = sp1_operation(h_op1_name, h_stride_1, h_filters)
    op2 = sp1_operation(h_op2_name, h_stride_2, h_filters)
    ops = [op1, op2]
    add_inputs, add_outputs = mi_add(len(ops))
    
    i_inputs = OrderedDict()
    for i in range(len(ops)):
        i_inputs['In' + str(i)] = ops[i][0]['In']
        add_inputs['In' + str(i)].connect(ops[i][1]['Out'])
    return (i_inputs, add_outputs)

# A module that selects an input from a list of inputs
def selector(h_selection, available, scope=None):
    def sub_fn(selection):
        ins, outs = available[selection]
        if len(ins) > 1:
            values = list(ins.values())
            ins = OrderedDict({'In': values[0]})
        return ins, outs
    return mo.substitution_module('Selector', {'selection': h_selection}, sub_fn, ['In'], ['Out'], scope)

def add_unused(h_selections, available, normal, name=None, scope=None):
    def dummy_func(**dummy_hs):
        return mo.empty()
    module = mo.SubstitutionModule('AddUnused', {'selection' + str(i): h_selections[i] for i in range(len(h_selections))}, dummy_func, ['In'], ['Out'], scope)
    
    def sub_fn(**selections):
        selected = [False] * len(selections)
        for k in selections:
            selected[selections[k]] = True

        # Disconnect first two factorized reductions if they are not used
        for i in range(2):
            if selected[i] and not normal:
                available[i][1]['Out'].module.inputs['In'].disconnect()
        
        unused = [available[i] for i in range(len(available)) if not selected[i]]
        ins, outs = mi_add(len(unused), name=name)
        
        module.inputs = {}
        for i in range(len(unused)):
            module._register_input('In' + str(i))
            unused[i][1]['Out'].connect(ins['In' + str(i)])
        
        return ins, outs
    module._substitution_fn = sub_fn
    return module.get_io()

def pad_and_shift():
    def cfn(di, dh):
        pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
        def fn(di):
            return {'Out': tf.pad(di['In'], pad_arr)[:, 1:, 1:, :]}
        return fn
    return htf.siso_tensorflow_module('Pad', cfn, {})

def concat(axis):
    def cfn(di, dh):
        def fn(di):
            return {'Out': tf.concat(values=[di['In0'], di['In1']], axis=axis)}
        return fn
    return TFM('Concat', {}, cfn, ['In0', 'In1'], ['Out']).get_io()

def factorized_reduction(h_num_filters, h_stride):
    is_stride_2 = co.DependentHyperparameter(lambda stride: int(stride == 2), {'stride': h_stride})
    reduced_filters = co.DependentHyperparameter(lambda filters: filters / 2, {'filters': h_num_filters})
    ins, outs = mo.empty()
    path1_ins, path1_outs = mo.siso_sequential([
        avg_pool2d(D([1]), h_stride),
        conv2d(reduced_filters, D([1]), D([1]), D([1]), D([True]))
    ])
    path2_ins, path2_outs = mo.siso_sequential([
        pad_and_shift(),
        avg_pool2d(D([1]), h_stride),
        conv2d(reduced_filters, D([1]), D([1]), D([1]), D([True]))
    ])
    c_ins, c_outs = concat(3)
    path1_outs['Out'].connect(c_ins['In0'])
    outs['Out'].connect(path1_ins['In'])

    path2_outs['Out'].connect(c_ins['In1'])
    outs['Out'].connect(path2_ins['In'])

    return mo.siso_or([
        lambda: mo.siso_sequential([
            conv2d(h_num_filters, D([1]), h_stride, D([1]), D([True])),
            batch_normalization()
        ]),
        lambda: mo.siso_sequential([
            (ins, c_outs),
            batch_normalization()
        ])
    ], is_stride_2, name='factor')

# A module that outlines the basic cell structure in Learning Transferable
# Architectures for Scalable Image Recognition (Zoph et al, 2017)
def basic_cell(available, h_sharer, h_filters, C=5, normal=True):
    assert len(available) == 2
    i_inputs = OrderedDict([('In' + str(idx), available[idx][1]['Out']) for idx in range(len(available))])
    selections = []
    for i in range(C):
        if normal:
            h_op0 = h_sharer.get('h_norm_op0_' + str(i))
            h_op1 = h_sharer.get('h_norm_op1_' + str(i))
            h_in0_pos = h_sharer.get('h_norm_in0_pos_' + str(i))
            h_in1_pos = h_sharer.get('h_norm_in1_pos_' + str(i))
        else:
            h_op0 = h_sharer.get('h_red_op0_' + str(i))
            h_op1 = h_sharer.get('h_red_op1_' + str(i))
            h_in0_pos = h_sharer.get('h_red_in0_pos_' + str(i))
            h_in1_pos = h_sharer.get('h_red_in1_pos_' + str(i))

        selections.append(h_in0_pos)
        selections.append(h_in1_pos)
        sel0_inputs, sel0_outputs = selector(h_in0_pos, available[:])
        sel1_inputs, sel1_outputs = selector(h_in1_pos, available[:])

        def is_stride_2(pos):
            return 1 if normal or pos > 1 else 2 
        h_stride_0 = co.DependentHyperparameter(is_stride_2, {'pos': h_in0_pos}, name='h_stride_0')
        h_stride_1 = co.DependentHyperparameter(is_stride_2, {'pos': h_in1_pos}, name='h_stride_1')
        comb_inputs, comb_outputs = sp1_combine(
            h_op0, h_op1, h_stride_0, h_stride_1, h_filters)
        sel0_outputs['Out'].connect(comb_inputs['In0'])
        sel1_outputs['Out'].connect(comb_inputs['In1'])
        available.append((comb_inputs, comb_outputs))

    if not normal:
        available[0] = mo.siso_sequential([
            available[0], 
            factorized_reduction(h_filters, D([2]))
        ])
        available[1] = mo.siso_sequential([
            available[1],
            factorized_reduction(h_filters, D([2]))
        ])

    add_inputs, add_outputs = add_unused(selections, available[:], normal, 'NORMAL_CELL' if normal else 'REDUCTION_CELL')
    return i_inputs, add_outputs


# A module that creates a number of repeated cells (both reduction and normal)
# as in Learning Transferable Architectures for Scalable Image Recognition (Zoph et al, 2017)
def ss_repeat(input_layers, h_N, h_sharer, h_filters, C, num_ov_repeat, num_classes, scope=None):
    def sub_fn(N):
        assert N > 0

        h_iter = h_filters
        for i in range(num_ov_repeat):
            for j in range(N):
                available = input_layers[-2:]
                norm_cell = basic_cell(available, h_sharer, h_iter, C=C, normal=True)
                input_layers.append(norm_cell)
            if i < num_ov_repeat - 1:
                h_iter = co.DependentHyperparameter(
                    lambda filters: filters*2,
                    {'filters': h_iter}, name='h_iter')
                available = input_layers[-2:]
                red_cell = basic_cell(available, h_sharer, h_iter, C=C, normal=False)
                input_layers[-1] = mo.siso_sequential([
                    input_layers[-1],
                    factorized_reduction(h_iter, D([2]))
                ])
                input_layers.append(red_cell)
                
        return mo.siso_sequential([
            input_layers[-1],
            global_pool2d(),
            fc_layer(D([num_classes]))
        ])
    return mo.substitution_module('SS_repeat', {'N': h_N}, sub_fn, ['In0', 'In1'], ['Out'], scope)

# Creates search space using operations from search spaces 1 and 3 from
# Regularized Evolution for Image Classifier Architecture Search (Real et al, 2018)
def get_search_space_small(num_classes, C):
    co.Scope.reset_default_scope()
    C = 5
    h_N = D([3])
    h_F = D([24])
    h_sharer = hp.HyperparameterSharer()
    for i in range(C):
        h_sharer.register('h_norm_op0_' + str(i), lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 'dil3_2', 's_sep7'], name='Mutatable'))
        h_sharer.register('h_norm_op1_' + str(i), lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 'dil3_2', 's_sep7'], name='Mutatable'))
        h_sharer.register('h_norm_in0_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
        h_sharer.register('h_norm_in1_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
    for i in range(C):
        h_sharer.register('h_red_op0_' + str(i), lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 's_sep7'], name='Mutatable'))
        h_sharer.register('h_red_op1_' + str(i), lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 's_sep7'], name='Mutatable'))
        h_sharer.register('h_red_in0_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
        h_sharer.register('h_red_in1_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
    i_inputs, i_outputs = conv2d(h_F, D([1]), D([1]), D([1]), D([True]))
    o_inputs, o_outputs = ss_repeat([(i_inputs, i_outputs), (i_inputs, i_outputs)], h_N, h_sharer, h_F, C, 3, num_classes)
    return i_inputs, o_outputs

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
    h_F = D([24])
    h_sharer = hp.HyperparameterSharer()
    for i in range(C):
        h_sharer.register('h_norm_op0_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
          'min2', 'max2', 'max3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6'], name='Mutatable'))
        h_sharer.register('h_norm_op1_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
          'min2', 'max2', 'max3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6'], name='Mutatable'))
        h_sharer.register('h_in0_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
        h_sharer.register('h_in1_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
    for i in range(C):
        if i == 0:
            h_sharer.register('h_red_op0_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
            'min2', 'max2', 'max3', 's_sep3' 's_sep7'], name='Mutatable'))
            h_sharer.register('h_red_op1_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
            'min2', 'max2', 'max3', 's_sep3' 's_sep7'], name='Mutatable'))
        else:
            h_sharer.register('h_red_op0_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
            'min2', 'max2', 'max3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6'], name='Mutatable'))
            h_sharer.register('h_red_op1_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
            'min2', 'max2', 'max3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6'], name='Mutatable'))
        h_sharer.register('h_red_in0_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
        h_sharer.register('h_red_in1_pos_' + str(i), lambda i=i: D(list(range(2 + i)), name='Mutatable'))
    i_inputs, i_outputs = conv2d(h_F, D([1]), D([1]), D([1]), D([True]))
    o_inputs, o_outputs = ss_repeat([(i_inputs, i_outputs), (i_inputs, i_outputs)], h_N, h_sharer, h_F, C, 3, num_classes)
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
