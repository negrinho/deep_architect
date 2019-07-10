from builtins import str
import tensorflow as tf
import deep_architect.modules as mo
from deep_architect.helpers import tfeager_support as htfe

TFEM = htfe.TFEModule


def avg_pool(h_kernel_size, h_stride):

    def compile_fn(di, dh):

        def fn(di, isTraining=True):
            with tf.device('/gpu:0'):
                return {
                    'Out':
                    tf.nn.avg_pool(di['In'],
                                   [1, dh['kernel_size'], dh['kernel_size'], 1],
                                   [1, dh['stride'], dh['stride'], 1], 'SAME')
                }

        return fn

    return htfe.siso_tfeager_module('AvgPool', compile_fn, {
        'kernel_size': h_kernel_size,
        'stride': h_stride,
    })


def max_pool(h_kernel_size, h_stride):

    def compile_fn(di, dh):

        def fn(di, isTraining=True):
            with tf.device('/gpu:0'):
                return {
                    'Out':
                    tf.nn.max_pool(di['In'],
                                   [1, dh['kernel_size'], dh['kernel_size'], 1],
                                   [1, dh['stride'], dh['stride'], 1], 'SAME')
                }

        return fn

    return htfe.siso_tfeager_module('MaxPool2D', compile_fn, {
        'kernel_size': h_kernel_size,
        'stride': h_stride,
    })


def keras_batch_normalization(name='default', weight_sharer=None):
    name = name + '_bn'

    def compile_fn(di, dh):
        bn = weight_sharer.get(name, tf.keras.layers.BatchNormalization, lambda
                               layer: layer.get_weights())
        if not bn.built:
            with tf.device('/gpu:0'):
                bn.build(di['In'].get_shape())
                weights = weight_sharer.load_weights(name)
                if weights is not None:
                    bn.set_weights(weights)

        def fn(di, isTraining):
            with tf.device('/gpu:0'):
                return {'Out': bn(di['In'], training=isTraining)}

        return fn

    return htfe.siso_tfeager_module('BatchNormalization', compile_fn, {})


def relu():

    def compile_fn(di, dh):

        def fn(di, isTraining=True):
            with tf.device('/gpu:0'):
                return {'Out': tf.nn.relu(di['In'])}

        return fn

    return htfe.siso_tfeager_module('ReLU', compile_fn, {})


def conv2D(filter_size, name, weight_sharer, out_filters=None):

    def compile_fn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        channels = channels if out_filters is None else out_filters

        conv_fn = lambda: tf.keras.layers.Conv2D(
            channels, filter_size, padding='same')
        conv = weight_sharer.get(name + '_conv_' + str(filter_size),
                                 conv_fn, lambda layer: layer.get_weights())
        if not conv.built:
            with tf.device('/gpu:0'):
                conv.build(di['In'].get_shape())
                weights = weight_sharer.load_weights(name)
                if weights is not None:
                    conv.set_weights(weights)

        def fn(di, isTraining=True):
            with tf.device('/gpu:0'):
                return {'Out': conv(di['In'])}

        return fn

    return htfe.siso_tfeager_module('Conv2D', compile_fn, {})


def conv2D_depth_separable(filter_size, name, weight_sharer, out_filters=None):

    def compile_fn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        channels = channels if out_filters is None else out_filters
        conv_fn = lambda: tf.keras.layers.SeparableConv2D(
            channels, filter_size, padding='same')
        conv = weight_sharer.get(name + '_dsep_' + str(filter_size),
                                 conv_fn, lambda layer: layer.get_weights())
        if not conv.built:
            with tf.device('/gpu:0'):
                conv.build(di['In'].get_shape())
                weights = weight_sharer.load_weights(name)
                if weights is not None:
                    conv.set_weights(weights)

        def fn(di, isTraining=True):
            with tf.device('/gpu:0'):
                return {'Out': conv(di['In'])}

        return fn

    return htfe.siso_tfeager_module('Conv2DSeparable', compile_fn, {})


def global_pool():

    def compile_fn(di, dh):

        def fn(di, isTraining):
            with tf.device('/gpu:0'):
                return {'Out': tf.reduce_mean(di['In'], [1, 2])}

        return fn

    return htfe.siso_tfeager_module('GlobalPool', compile_fn, {})


def dropout(keep_prob):

    def compile_fn(di, dh):

        def fn(di, isTraining=True):
            if isTraining:
                with tf.device('/gpu:0'):
                    out = tf.nn.dropout(di['In'], keep_prob)
            else:
                out = di['In']
            return {'Out': out}

        return fn

    return htfe.siso_tfeager_module('Dropout', compile_fn, {})


def fc_layer(num_classes, name, weight_sharer):
    name = name + '_fc_layer_' + str(num_classes)

    def compile_fn(di, dh):
        fc = weight_sharer.get(name, lambda: tf.keras.layers.Dense(num_classes),
                               lambda layer: layer.get_weights())
        if not fc.built:
            with tf.device('/gpu:0'):
                fc.build(di['In'].get_shape())
                weights = weight_sharer.load_weights(name)
                if weights is not None:
                    fc.set_weights(weights)

        def fn(di, isTraining=True):
            with tf.device('/gpu:0'):
                return {'Out': fc(di['In'])}

        return fn

    return htfe.siso_tfeager_module('FC_Layer', compile_fn, {})


def wrap_relu_batch_norm(io_pair,
                         add_relu=True,
                         add_bn=True,
                         weight_sharer=None,
                         name=None):
    assert add_relu or add_bn
    elements = [True, add_relu, add_bn]
    module_fns = [
        lambda: io_pair,
        relu, lambda: keras_batch_normalization(name=name,
                                                weight_sharer=weight_sharer)
    ]
    return mo.siso_sequential(
        [module_fn() for i, module_fn in enumerate(module_fns) if elements[i]])


def wrap_batch_norm_relu(io_pair,
                         add_relu=True,
                         add_bn=True,
                         weight_sharer=None,
                         name=None):
    assert add_relu or add_bn
    elements = [True, add_bn, add_relu]
    module_fns = [
        lambda: io_pair, lambda: keras_batch_normalization(
            name=name, weight_sharer=weight_sharer), relu
    ]
    return mo.siso_sequential(
        [module_fn() for i, module_fn in enumerate(module_fns) if elements[i]])
