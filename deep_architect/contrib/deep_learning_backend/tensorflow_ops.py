from deep_architect.helpers.tensorflow_support import (siso_tensorflow_module,
                                                       TensorflowModule)
import tensorflow as tf


def conv2d(h_num_filters,
           h_filter_width,
           h_stride=1,
           h_dilation_rate=1,
           h_use_bias=True,
           h_padding='SAME'):

    def compile_fn(di, dh):
        conv_op = tf.layers.Conv2D(dh['num_filters'],
                                   dh['filter_width'],
                                   dh['stride'],
                                   use_bias=dh['use_bias'],
                                   dilation_rate=dh['dilation_rate'],
                                   padding=dh['padding'])

        def fn(di):
            return {'out': conv_op(di['in'])}

        return fn

    return siso_tensorflow_module(
        'Conv2D', compile_fn, {
            'num_filters': h_num_filters,
            'filter_width': h_filter_width,
            'stride': h_stride,
            'use_bias': h_use_bias,
            'dilation_rate': h_dilation_rate,
            'padding': h_padding
        })


def separable_conv2d(h_num_filters,
                     h_filter_width,
                     h_stride=1,
                     h_dilation_rate=1,
                     h_depth_multiplier=1,
                     h_use_bias=True,
                     h_padding='SAME'):

    def compile_fn(di, dh):
        conv_op = tf.layers.SeparableConv2D(
            dh['num_filters'],
            dh['filter_width'],
            strides=dh['stride'],
            dilation_rate=dh['dilation_rate'],
            depth_multiplier=dh['depth_multiplier'],
            use_bias=dh['use_bias'],
            padding=dh['padding'])

        def fn(di):
            return {'out': conv_op(di['in'])}

        return fn

    return siso_tensorflow_module(
        'SeparableConv2D', compile_fn, {
            'num_filters': h_num_filters,
            'filter_width': h_filter_width,
            'stride': h_stride,
            'use_bias': h_use_bias,
            'dilation_rate': h_dilation_rate,
            'depth_multiplier': h_depth_multiplier,
            'padding': h_padding
        })


def max_pool2d(h_kernel_size, h_stride=1):

    def compile_fn(di, dh):

        def fn(di):
            return {
                'out':
                tf.nn.max_pool(di['in'],
                               [1, dh['kernel_size'], dh['kernel_size'], 1],
                               [1, dh['stride'], dh['stride'], 1], 'SAME')
            }

        return fn

    return siso_tensorflow_module('MaxPool2D', compile_fn, {
        'kernel_size': h_kernel_size,
        'stride': h_stride,
    })


def avg_pool2d(h_kernel_size, h_stride=1):

    def compile_fn(di, dh):

        def fn(di):
            return {
                'out':
                tf.nn.avg_pool(di['in'],
                               [1, dh['kernel_size'], dh['kernel_size'], 1],
                               [1, dh['stride'], dh['stride'], 1], 'SAME')
            }

        return fn

    return siso_tensorflow_module('MaxPool2D', compile_fn, {
        'kernel_size': h_kernel_size,
        'stride': h_stride,
    })


def dropout(h_keep_prob):

    def compile_fn(di, dh):
        p = tf.placeholder(tf.float32)

        def fn(di):
            return {'out': tf.nn.dropout(di['in'], p)}

        return fn, {p: dh['keep_prob']}, {p: 1.0}

    return siso_tensorflow_module('Dropout', compile_fn,
                                  {'keep_prob': h_keep_prob})


def batch_normalization():

    def compile_fn(di, dh):
        p_var = tf.placeholder(tf.bool)

        def fn(di):
            return {
                'out': tf.layers.batch_normalization(di['in'], training=p_var)
            }

        return fn, {p_var: 1}, {p_var: 0}

    return siso_tensorflow_module('BatchNormalization', compile_fn, {})


def relu():
    return siso_tensorflow_module(
        'ReLU', lambda di, dh: lambda di: {'out': tf.nn.relu(di['in'])}, {})


def add():
    return TensorflowModule('Add', lambda: lambda In0, In1: tf.add(In0, In1),
                            {}, ['in0', 'in1'], ['out']).get_io()


def global_pool2d():

    def compile_fn(di, dh):

        def fn(di):
            return {'out': tf.reduce_mean(di['in'], [1, 2])}

        return fn

    return siso_tensorflow_module('GlobalAveragePool', compile_fn, {})


def fc_layer(h_num_units):

    def compile_fn(di, dh):
        fc = tf.layers.Dense(dh['num_units'])

        def fn(di):
            return {'out': fc(di['in'])}

        return fn

    return siso_tensorflow_module('FCLayer', compile_fn,
                                  {'num_units': h_num_units})


func_dict = {
    'dropout': dropout,
    'conv2d': conv2d,
    'max_pool2d': max_pool2d,
    'avg_pool2d': avg_pool2d,
    'batch_normalization': batch_normalization,
    'relu': relu,
    'global_pool2d': global_pool2d,
    'fc_layer': fc_layer
}