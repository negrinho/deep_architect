import tensorflow as tf
from darch.contrib.search_spaces.tensorflow.common import D, siso_tfm
import darch.contrib.search_spaces.tensorflow.dnn as dnn
import darch.modules as mo
import numpy as np

def kaiming2015delving_initializer_conv(gain=1.0):
    def init_fn(shape):
        n = np.product(shape)
        stddev = gain * np.sqrt(2.0 / n)
        init_vals = tf.random_normal(shape, 0.0, stddev)
        return init_vals
    return init_fn

def conv2d(h_num_filters, h_filter_width, h_stride, h_use_bias, h_W_init_fn, h_b_init_fn):
    def cfn(di, dh):
        (_, _, _, num_channels) = di['In'].get_shape().as_list()
        W = tf.Variable(dh['W_init_fn']([dh['filter_width'], dh['filter_width'], num_channels, dh['num_filters']]))
        if dh['use_bias']:
            b = tf.Variable(dh['b_init_fn']([dh['num_filters']]))
        def fn(di):
            out = tf.nn.conv2d(di['In'], W, [1, dh['stride'], dh['stride'], 1], 'SAME')
            if dh['use_bias']:
                out = tf.nn.bias_add(out, b)
            return {'Out' : out}
        return fn
    return siso_tfm('Conv2D', cfn, {
        'num_filters' : h_num_filters,
        'filter_width' : h_filter_width,
        'stride' : h_stride,
        'use_bias' : h_use_bias,
        'W_init_fn' : h_W_init_fn,
        'b_init_fn' : h_b_init_fn,
        })

def conv2d_simplified(h_num_filters, h_filter_width, stride, use_bias):
    W_init_fn = kaiming2015delving_initializer_conv()
    b_init_fn = dnn.constant_initializer(0.0)
    return conv2d(h_num_filters, h_filter_width,
        D([stride]), D([use_bias]), D([W_init_fn]), D([b_init_fn]))

def max_pool2d(h_kernel_size, h_stride):
    def cfn(di, dh):
        def fn(di):
            return {'Out' : tf.nn.max_pool(di['In'],
                [1, dh['kernel_size'], dh['kernel_size'], 1], [1, dh['stride'], dh['stride'], 1], 'SAME')}
        return fn
    return siso_tfm('MaxPool2D', cfn, {
        'kernel_size' : h_kernel_size, 'stride' : h_stride,})

def conv_cell(h_num_filters, h_filter_width, h_swap, h_opt_drop, h_keep_prob, stride):
    assert stride >= 1
    return mo.siso_sequential([
        conv2d_simplified(h_num_filters, h_filter_width, stride, 0),
        mo.siso_permutation([dnn.relu, dnn.batch_normalization], h_swap),
        mo.siso_optional(lambda: dnn.dropout(h_keep_prob), h_opt_drop)])

def conv_net(h_num_spatial_reductions):
    h_swap = D([0, 1])
    h_opt_drop = D([0, 1])
    h_keep_prob = D([0.25, 0.5, 0.75, 1.0])

    get_conv_cell = lambda stride: conv_cell(
        D([32, 64, 128, 256, 512]), D([1, 3, 5]),
        h_swap, h_opt_drop, h_keep_prob, stride)

    get_reduction_cell = lambda: get_conv_cell(2)
    get_normal_cell = lambda: get_conv_cell(1)

    return mo.siso_repeat(
        lambda: mo.siso_sequential([
            get_reduction_cell(),
            mo.siso_repeat(get_normal_cell, D([1, 2, 4, 8])),
        ]), h_num_spatial_reductions)

def spatial_squeeze(h_pool_op, h_num_hidden):
    def cfn(di, dh):
        (_, height, width, num_channels) = di['In'].get_shape().as_list()

        W_init_fn = kaiming2015delving_initializer_conv()
        b_init_fn = dnn.constant_initializer(0.0)
        W = tf.Variable(W_init_fn([1, 1, num_channels, dh['num_hidden']]))
        b = tf.Variable(b_init_fn([dh['num_hidden']]))
        def fn(di):
            out = tf.nn.conv2d(di['In'], W, [1, 1, 1, 1], 'SAME')
            out = tf.nn.bias_add(out, b)
            if dh['pool_op'] == 'max' or dh['pool_op'] == 'avg':
                out = tf.nn.pool(out, [height, width], dh['pool_op'].upper(), 'VALID')
                assert tuple(out.get_shape().as_list()[1:3]) == (1, 1)
            else:
                raise ValueError
            out = tf.squeeze(out, [1, 2])
            return {'Out' : out}
        return fn
    return siso_tfm('SpatialSqueeze', cfn, {
        'pool_op' : h_pool_op, 'num_hidden' : h_num_hidden})
