from __future__ import absolute_import
from builtins import str
from builtins import range
import deep_architect.core as co
import deep_architect.hyperparameters as hp
import deep_architect.helpers.tensorflow as htf
import deep_architect.modules as mo
import deep_architect.contrib.misc.search_spaces.tensorflow.cnn2d as cnn2d
from .common_ops import relu, add, pool_and_logits, batch_normalization, avg_pool, wrap_relu_batch_norm
from deep_architect.contrib.misc.search_spaces.tensorflow.common import D, siso_tensorflow_module
import tensorflow as tf
import numpy as np


TFM = htf.TensorflowModule
const_fn = lambda c: lambda shape: tf.constant(c, shape=shape)

# Basic convolutional module that selects number of filters based on number of
# input channels and the stride as in "Learning transferable architectures for scalable
# image recognition" (Zoph et al, 2017)
def conv2D(h_filter_size, h_stride):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        W_init_fn = cnn2d.kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        num_filters = 2 * channels if dh['stride'] == 2 else channels

        W = tf.Variable( W_init_fn( [dh['filter_size'], dh['filter_size'], channels, num_filters] ) )
        b = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(di):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(di['In'], W, [1, dh['stride'], dh['stride'], 1], 'SAME'), b)}
        return fn

    return siso_tensorflow_module('Conv2D', cfn, {
        'stride' : h_stride,
        'filter_size' : h_filter_size})

# Spatially separable convolutions, with output filter calculations as in Zoph17
def conv_spatial_separable(h_filter_size, h_stride):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        W_init_fn = cnn2d.kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        num_filters = 2 * channels if dh['stride'] == 2 else channels

        W1 = tf.Variable( W_init_fn( [1, dh['filter_size'], channels, num_filters] ) )
        b1 = tf.Variable( b_init_fn( [num_filters] ) )
        W2 = tf.Variable( W_init_fn( [dh['filter_size'], 1, num_filters, num_filters] ) )
        b2 = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(di):
            intermediate = tf.nn.bias_add(
                tf.nn.conv2d(di['In'], W1, [1, dh['stride'], dh['stride'], 1], 'SAME'), b1)
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(intermediate, W2, [1, 1, 1, 1], 'SAME'), b2)}
        return fn

    return siso_tensorflow_module('Conv2DSimplified', cfn, {
        'filter_size' : h_filter_size,
        'stride' : h_stride})


# Dilated convolutions, with output filter calculations as in Zoph17
def conv2D_dilated(h_filter_size, h_dilation):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        W_init_fn = cnn2d.kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        W = tf.Variable( W_init_fn( [dh['filter_size'], dh['filter_size'], channels, channels] ) )
        b = tf.Variable( b_init_fn( [channels] ) )
        def fn(di):
            return {'Out' : tf.nn.bias_add(
                tf.nn.atrous_conv2d(di['In'], W, dh['dilation'], 'SAME'), b)}
        return fn

    return siso_tensorflow_module('Conv2DDilated', cfn, {
        'filter_size' : h_filter_size,
        'dilation' : h_dilation
        })


# Depth separable convolutions, with output filter calculations as in Zoph17
def conv2D_depth_separable(h_filter_size, h_channel_multiplier, h_stride):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        W_init_fn = cnn2d.kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        num_filters = 2 * channels if dh['stride'] == 2 else channels

        W_depth = tf.Variable( W_init_fn( [dh['filter_size'], dh['filter_size'], channels, dh['channel_multiplier']] ) )
        W_point = tf.Variable( W_init_fn( [1, 1, channels * dh['channel_multiplier'], num_filters] ) )
        b = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(di):
            return {'Out' : tf.nn.bias_add(
                tf.nn.separable_conv2d(di['In'], W_depth, W_point, [1, dh['stride'], dh['stride'], 1], 'SAME', [1, 1]), b)}
        return fn

    return siso_tensorflow_module('Conv2DSeparable', cfn, {
        'filter_size' : h_filter_size,
        'channel_multiplier' : h_channel_multiplier,
        'stride' : h_stride})

def conv1x1(h_num_filters):
    return cnn2d.conv2d(h_num_filters, D([1]), D([1]), D([True]))

# The operations used in search space 1 and search space 3 in Regularized Evolution for
# Image Classifier Architecture Search (Real et al, 2018)
def sp1_operation(h_op_name, h_stride):
    return mo.siso_or({
            'identity': lambda: mo.empty(),
            'd_sep3': lambda: wrap_relu_batch_norm(conv2D_depth_separable(D([3]), D([1]), h_stride)),
            'd_sep5': lambda: wrap_relu_batch_norm(conv2D_depth_separable(D([5]), D([1]), h_stride)),
            'd_sep7': lambda: wrap_relu_batch_norm(conv2D_depth_separable(D([7]), D([1]), h_stride)),
            'avg3': lambda: avg_pool(D([3]), h_stride),
            'max3': lambda: cnn2d.max_pool2d(D([3]), h_stride),
            'dil3': lambda: wrap_relu_batch_norm(conv2D_dilated(D([3]), D([2]))),
            's_sep7': lambda: wrap_relu_batch_norm(conv_spatial_separable(D([7]), h_stride))
            }, h_op_name)

# The operations used in search space 2 in Regularized Evolution for
# Image Classifier Architecture Search (Real et al, 2018)
def sp2_operation(h_op_name, h_stride):
    return mo.siso_or({
            'identity': lambda: mo.empty(),
            'conv1': lambda: wrap_relu_batch_norm(conv2D(D([1]), h_stride)),
            'conv3': lambda: wrap_relu_batch_norm(conv2D(D([3]), h_stride)),
            'd_sep3': lambda: wrap_relu_batch_norm(conv2D_depth_separable(D([3]), D([1]), h_stride)),
            'd_sep5': lambda: wrap_relu_batch_norm(conv2D_depth_separable(D([5]), D([1]), h_stride)),
            'd_sep7': lambda: wrap_relu_batch_norm(conv2D_depth_separable(D([7]), D([1]), h_stride)),
            'avg2': lambda: avg_pool(D([2]), h_stride),
            'avg3': lambda: avg_pool(D([3]), h_stride),
            'min2': lambda: mo.empty(),
            'max2': lambda: max_pool(D([2]), h_stride),
            'max3': lambda: max_pool(D([3]), h_stride),
            'dil3': lambda: wrap_relu_batch_norm(conv2D_dilated(D([3]), D([2]))),
            'dil5': lambda: wrap_relu_batch_norm(conv2D_dilated(D([5]), D([2]))),
            'dil7': lambda: wrap_relu_batch_norm(conv2D_dilated(D([7]), D([2]))),
            's_sep3': lambda: wrap_relu_batch_norm(conv_spatial_separable(D([3]), h_stride)),
            's_sep7': lambda: wrap_relu_batch_norm(conv_spatial_separable(D([7]), h_stride)),
            'dil3_2': lambda: wrap_relu_batch_norm(conv2D_dilated(D([3]), D([2]))),
            'dil3_4': lambda: wrap_relu_batch_norm(conv2D_dilated(D([3]), D([4]))),
            'dil3_6': lambda: wrap_relu_batch_norm(conv2D_dilated(D([3]), D([6])))
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
def sp1_combine(h_op1_name, h_op2_name, h_stride):
    in1 = lambda: sp1_operation(h_op1_name, h_stride)
    in2 = lambda: sp1_operation(h_op2_name, h_stride)
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
def basic_cell(h_sharer, C=5, normal=True):
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

        comb_inputs, comb_outputs = sp1_combine(h_op1, h_op2, D([1 if normal or i > 0 else 2]))
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
def ss_repeat(h_N, h_sharer, C, num_ov_repeat, num_classes, scope=None):
    def sub_fn(N):
        assert N > 0
        i_inputs, i_outputs = mo.simo_split(2)
        prev_2 = [i_outputs['Out0'], i_outputs['Out1']]

        for i in range(num_ov_repeat):
            for j in range(N):
                norm_inputs, norm_outputs = basic_cell(h_sharer, C=C, normal=True)
                prev_2[0].connect(norm_inputs['In0'])
                prev_2[1].connect(norm_inputs['In1'])
                prev_2[0] = prev_2[1]
                prev_2[1] = norm_outputs['Out']
            if j < num_ov_repeat - 1:
                red_inputs, red_outputs = basic_cell(h_sharer, C=C, normal=False)
                prev_2[0].connect(red_inputs['In0'])
                prev_2[1].connect(red_inputs['In1'])
                prev_2[0] = prev_2[1]
                prev_2[1] = red_outputs['Out']

        logits_inputs, logits_outputs = pool_and_logits(num_classes)
        prev_2[1].connect(logits_inputs['In'])
        return i_inputs, logits_outputs
    return mo.substitution_module('SS_repeat', {'N': h_N}, sub_fn, ['In'], ['Out'], scope)

# Creates search space using operations from search spaces 1 and 3 from
# Regularized Evolution for Image Classifier Architecture Search (Real et al, 2018)
def get_search_space_small(num_classes, C):
    co.Scope.reset_default_scope()
    C = 5
    h_N = D([3])
    h_F = D([8])
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
    i_inputs, i_outputs = conv1x1(h_F)
    o_inputs, o_outputs = ss_repeat(h_N, h_sharer, C, 3, num_classes)
    i_outputs['Out'].connect(o_inputs['In'])
    r_inputs, r_outputs = mo.siso_sequential([mo.empty(), (i_inputs, o_outputs), mo.empty()])
    return r_inputs, r_outputs

# Search space 1 from Regularized Evolution for Image Classifier Architecture Search (Real et al, 2018)
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
