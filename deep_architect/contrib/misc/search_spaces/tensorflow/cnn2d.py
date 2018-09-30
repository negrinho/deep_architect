from deep_architect.contrib.misc.search_spaces.tensorflow.common import D, siso_tensorflow_module
import deep_architect.contrib.misc.search_spaces.tensorflow.dnn as dnn
import deep_architect.modules as mo
import tensorflow as tf
import numpy as np


def kaiming2015delving_initializer_conv(gain=1.0):

    def init_fn(shape):
        n = np.product(shape)
        stddev = gain * np.sqrt(2.0 / n)
        init_vals = tf.random_normal(shape, 0.0, stddev)
        return init_vals

    return init_fn


def conv2d(h_num_filters, h_filter_width, h_stride, h_use_bias):

    def compile_fn(di, dh):
        conv_op = tf.layers.Conv2D(
            dh['num_filters'], (dh['filter_width'],) * 2, (dh['stride'],) * 2,
            use_bias=dh['use_bias'],
            padding='SAME')

        def forward_fn(di):
            return {'Out': conv_op(di['In'])}

        return forward_fn

    return siso_tensorflow_module(
        'Conv2D', compile_fn, {
            'num_filters': h_num_filters,
            'filter_width': h_filter_width,
            'stride': h_stride,
            'use_bias': h_use_bias,
        })


def max_pool2d(h_kernel_size, h_stride):

    def compile_fn(di, dh):

        def forward_fn(di):
            return {
                'Out':
                tf.nn.max_pool(di['In'],
                               [1, dh['kernel_size'], dh['kernel_size'], 1],
                               [1, dh['stride'], dh['stride'], 1], 'SAME')
            }

        return forward_fn

    return siso_tensorflow_module('MaxPool2D', compile_fn, {
        'kernel_size': h_kernel_size,
        'stride': h_stride,
    })


def conv_cell(h_num_filters, h_filter_width, h_swap, h_opt_drop, h_keep_prob,
              stride):
    assert stride >= 1
    return mo.siso_sequential([
        conv2d(h_num_filters, h_filter_width, D([stride]), D([0])),
        mo.siso_permutation([dnn.relu, dnn.batch_normalization], h_swap),
        mo.siso_optional(lambda: dnn.dropout(h_keep_prob), h_opt_drop)
    ])


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
            mo.siso_repeat(get_normal_cell, D([1, 2, 4, 8])),]),
        h_num_spatial_reductions)


def spatial_squeeze(h_pool_op, h_num_hidden):

    def compile_fn(di, dh):
        (_, height, width, _) = di['In'].get_shape().as_list()
        conv_op = tf.layers.Conv2D(dh['num_hidden'], (1, 1), (1, 1))

        def forward_fn(di):
            out = conv_op(di['In'])
            if dh['pool_op'] == 'max' or dh['pool_op'] == 'avg':
                out = tf.nn.pool(out, [height, width], dh['pool_op'].upper(),
                                 'VALID')
                assert tuple(out.get_shape().as_list()[1:3]) == (1, 1)
            else:
                raise ValueError
            out = tf.squeeze(out, [1, 2])
            return {'Out': out}

        return forward_fn

    return siso_tensorflow_module('SpatialSqueeze', compile_fn, {
        'pool_op': h_pool_op,
        'num_hidden': h_num_hidden
    })
