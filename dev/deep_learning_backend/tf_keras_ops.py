from dev.helpers.tfeager import siso_tfeager_module, TFEModule
from deep_architect.hyperparameters import D
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops


def max_pool2d(h_kernel_size, h_stride=1, h_padding='SAME'):

    def compile_fn(di, dh):
        pool = tf.keras.layers.MaxPooling2D(dh['kernel_size'],
                                            dh['stride'],
                                            padding=dh['padding'])

        def forward_fn(di, isTraining=True):
            return {'Out': pool(di['In'])}

        return forward_fn

    return siso_tfeager_module('MaxPool2D', compile_fn, {
        'kernel_size': h_kernel_size,
        'stride': h_stride,
        'padding': h_padding
    })


def min_pool2d(h_kernel_size, h_stride=1, h_padding='SAME'):

    def compile_fn(di, dh):
        pool = tf.keras.layers.MaxPooling2D(dh['kernel_size'],
                                            dh['stride'],
                                            padding=dh['padding'])
        negate = tf.keras.layers.Lambda(lambda x: -1 * x, name='negate')

        def forward_fn(di, isTraining=True):
            return {'Out': negate(pool(negate(di['In'])))}

        return forward_fn

    return siso_tfeager_module('MinPool2D', compile_fn, {
        'kernel_size': h_kernel_size,
        'stride': h_stride,
        'padding': h_padding
    })


def avg_pool2d(h_kernel_size, h_stride=1, h_padding='SAME'):

    def compile_fn(di, dh):
        pool = tf.keras.layers.AveragePooling2D(dh['kernel_size'],
                                                dh['stride'],
                                                padding=dh['padding'])

        def forward_fn(di, isTraining=True):

            return {'Out': pool(di['In'])}

        return forward_fn

    return siso_tfeager_module('MaxPool2D', compile_fn, {
        'kernel_size': h_kernel_size,
        'stride': h_stride,
        'padding': h_padding
    })


def batch_normalization():

    def compile_fn(di, dh):
        bn = tf.keras.layers.BatchNormalization(momentum=.9, epsilon=1e-5)

        def forward_fn(di, isTraining):
            return {'Out': bn(di['In'])}

        return forward_fn

    return siso_tfeager_module('BatchNormalization', compile_fn, {})


def relu():

    def compile_fn(di, dh):
        relu = tf.keras.layers.ReLU()

        def forward_fn(di, isTraining=True):
            return {'Out': relu(di['In'])}

        return forward_fn

    return siso_tfeager_module('ReLU', compile_fn, {})


def conv2d(h_num_filters,
           h_filter_width,
           h_stride=1,
           h_dilation_rate=1,
           h_use_bias=True,
           h_padding='SAME'):

    def compile_fn(di, dh):
        conv = tf.keras.layers.Conv2D(dh['num_filters'],
                                      dh['filter_width'],
                                      dh['stride'],
                                      use_bias=dh['use_bias'],
                                      dilation_rate=dh['dilation_rate'],
                                      padding=dh['padding'])

        def forward_fn(di, isTraining=True):
            return {'Out': conv(di['In'])}

        return forward_fn

    return siso_tfeager_module(
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

        conv_op = tf.keras.layers.SeparableConv2D(
            dh['num_filters'],
            dh['filter_width'],
            strides=dh['stride'],
            dilation_rate=dh['dilation_rate'],
            depth_multiplier=dh['depth_multiplier'],
            use_bias=dh['use_bias'],
            padding=dh['padding'])

        def fn(di, isTraining=True):
            return {'Out': conv_op(di['In'])}

        return fn

    return siso_tfeager_module(
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

    class Dropout(tf.keras.layers.Layer):

        def __init__(self, rate, seed=None, **kwargs):
            super(Dropout, self).__init__(**kwargs)
            self.rate = rate
            self.seed = seed
            self.supports_masking = True

        def call(self, inputs, training=None):
            if training is None:
                training = True

            def dropped_inputs():
                return tf.nn.dropout(inputs, 1 - self.rate, seed=self.seed)

            output = tf_utils.smart_cond(
                training, dropped_inputs, lambda: array_ops.identity(inputs))
            return output

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            config = {'rate': self.rate, 'seed': self.seed}
            base_config = super(Dropout, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    def compile_fn(di, dh):
        dropout_op = Dropout(1 - dh['keep_prob'])

        def forward_fn(di, isTraining=True):
            # out = tf.nn.dropout(di['In'], dh['keep_prob'])
            # else:
            #     out = di['In']
            out = dropout_op(di['In'])
            return {'Out': out}

        return forward_fn

    return siso_tfeager_module('Dropout', compile_fn,
                               {'keep_prob': h_keep_prob})


def global_pool2d():

    def compile_fn(di, dh):
        pool = tf.keras.layers.GlobalAveragePooling2D()

        def forward_fn(di, isTraining=True):
            return {'Out': pool(di['In'])}

        return forward_fn

    return siso_tfeager_module('GlobalAveragePool', compile_fn, {})


def flatten():

    def compile_fn(di, dh):
        flatten = tf.keras.layers.Flatten()

        def forward_fn(di, isTraining=True):
            return {'Out': flatten(di['In'])}

        return forward_fn

    return siso_tfeager_module('Flatten', compile_fn, {})


def fc_layer(h_num_units):

    def compile_fn(di, dh):
        fc = tf.keras.layers.Dense(dh['num_units'])

        def forward_fn(di, isTraining=True):
            return {'Out': fc(di['In'])}

        return forward_fn

    return siso_tfeager_module('FCLayer', compile_fn,
                               {'num_units': h_num_units})


def add(num_inputs):

    def compile_fn(di, dh):
        if num_inputs > 1:
            add = tf.keras.layers.Add()
        else:
            add = None

        def forward_fn(di, isTraining=True):
            out = add([di[inp] for inp in di]) if add else di['In0']

            return {'Out': out}

        return forward_fn

    return TFEModule('Add', {}, compile_fn,
                     ['In' + str(i) for i in range(num_inputs)],
                     ['Out']).get_io()


func_dict = {
    'dropout': dropout,
    'conv2d': conv2d,
    'max_pool2d': max_pool2d,
    'batch_normalization': batch_normalization,
    'relu': relu,
    'global_pool2d': global_pool2d,
    'fc_layer': fc_layer
}