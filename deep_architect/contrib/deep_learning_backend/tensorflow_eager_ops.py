from deep_architect.helpers.tensorflow_eager_support import siso_tensorflow_eager_module, TensorflowEagerModule
from deep_architect.hyperparameters import D
import tensorflow as tf


def max_pool2d(h_kernel_size, h_stride=1, h_padding='SAME'):

    def compile_fn(di, dh):
        pool = tf.layers.MaxPooling2D(dh['kernel_size'],
                                      dh['stride'],
                                      padding=dh['padding'])

        def forward_fn(di, is_training=True):
            return {'out': pool(di['in'])}

        return forward_fn

    return siso_tensorflow_eager_module('MaxPool2D', compile_fn, {
        'kernel_size': h_kernel_size,
        'stride': h_stride,
        'padding': h_padding
    })


def min_pool2d(h_kernel_size, h_stride=1, h_padding='SAME'):

    def compile_fn(di, dh):
        pool = tf.layers.MaxPooling2D(dh['kernel_size'],
                                      dh['stride'],
                                      padding=dh['padding'])

        def forward_fn(di, is_training=True):
            return {'out': -1 * pool(-1 * di['in'])}

        return forward_fn

    return siso_tensorflow_eager_module('MinPool2D', compile_fn, {
        'kernel_size': h_kernel_size,
        'stride': h_stride,
        'padding': h_padding
    })


def avg_pool2d(h_kernel_size, h_stride=1, h_padding='SAME'):

    def compile_fn(di, dh):
        pool = tf.layers.AveragePooling2D(dh['kernel_size'],
                                          dh['stride'],
                                          padding=dh['padding'])

        def forward_fn(di, is_training=True):
            return {'out': pool(di['in'])}

        return forward_fn

    return siso_tensorflow_eager_module('MaxPool2D', compile_fn, {
        'kernel_size': h_kernel_size,
        'stride': h_stride,
        'padding': h_padding
    })


def batch_normalization():

    def compile_fn(di, dh):
        bn = tf.layers.BatchNormalization(momentum=.9, epsilon=1e-5)

        def forward_fn(di, is_training):
            return {'out': bn(di['in'], training=is_training)}

        return forward_fn

    return siso_tensorflow_eager_module('BatchNormalization', compile_fn, {})


def relu():

    def compile_fn(di, dh):

        def forward_fn(di, is_training=True):
            return {'out': tf.nn.relu(di['in'])}

        return forward_fn

    return siso_tensorflow_eager_module('ReLU', compile_fn, {})


def conv2d(h_num_filters,
           h_filter_width,
           h_stride=1,
           h_dilation_rate=1,
           h_use_bias=True,
           h_padding='SAME'):

    def compile_fn(di, dh):
        conv = tf.layers.Conv2D(dh['num_filters'],
                                dh['filter_width'],
                                dh['stride'],
                                use_bias=dh['use_bias'],
                                dilation_rate=dh['dilation_rate'],
                                padding=dh['padding'])

        def forward_fn(di, is_training=True):
            return {'out': conv(di['in'])}

        return forward_fn

    return siso_tensorflow_eager_module(
        'Conv2D', compile_fn, {
            'num_filters': h_num_filters,
            'filter_width': h_filter_width,
            'stride': h_stride,
            'dilation_rate': h_dilation_rate,
            'use_bias': h_use_bias,
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

        def fn(di, is_training=True):
            return {'out': conv_op(di['in'])}

        return fn

    return siso_tensorflow_eager_module(
        'SeparableConv2D', compile_fn, {
            'num_filters': h_num_filters,
            'filter_width': h_filter_width,
            'stride': h_stride,
            'use_bias': h_use_bias,
            'dilation_rate': h_dilation_rate,
            'depth_multiplier': h_depth_multiplier,
            'padding': h_padding
        })


def dropout(h_keep_prob):

    def compile_fn(di, dh):

        def forward_fn(di, is_training=True):
            if is_training:
                out = tf.nn.dropout(di['in'], dh['keep_prob'])
            else:
                out = di['in']
            return {'out': out}

        return forward_fn

    return siso_tensorflow_eager_module('Dropout', compile_fn,
                                        {'keep_prob': h_keep_prob})


def global_pool2d():

    def compile_fn(di, dh):

        def forward_fn(di, is_training=True):
            return {'out': tf.reduce_mean(di['in'], [1, 2])}

        return forward_fn

    return siso_tensorflow_eager_module('GlobalAveragePool', compile_fn, {})


def flatten():

    def compile_fn(di, dh):

        def forward_fn(di, is_training=True):
            return {'out': tf.layers.flatten(di['in'])}

        return forward_fn

    return siso_tensorflow_eager_module('Flatten', compile_fn, {})


def fc_layer(h_num_units):

    def compile_fn(di, dh):
        fc = tf.layers.Dense(dh['num_units'])

        def forward_fn(di, is_training=True):
            return {'out': fc(di['in'])}

        return forward_fn

    return siso_tensorflow_eager_module('FCLayer', compile_fn,
                                        {'num_units': h_num_units})


def add(num_inputs):

    def compile_fn(di, dh):

        def forward_fn(di, is_training=True):
            out = tf.add_n([di[inp] for inp in di
                           ]) if len(di) > 1 else di['in0']

            return {'out': out}

        return forward_fn

    return TensorflowEagerModule('Add', compile_fn, {},
                                 ['in' + str(i) for i in range(num_inputs)],
                                 ['out']).get_io()


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