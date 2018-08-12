from builtins import str
import tensorflow as tf
import darch.modules as mo
from dev.architecture_search_benchmarks.helpers import tfeager as htfe
from dev.architecture_search_benchmarks.search_spaces.common_eager import siso_tfem

TFEM = htfe.TFEModule

def avg_pool(h_kernel_size, h_stride):
    def cfn(di, dh):
        def fn(di, isTraining=True):
            with tf.device('/gpu:0'):
                return {'Out' : tf.nn.avg_pool(di['In'],
                    [1, dh['kernel_size'], dh['kernel_size'], 1], [1, dh['stride'], dh['stride'], 1], 'SAME')}
        return fn
    return siso_tfem('AvgPool', cfn, {
        'kernel_size' : h_kernel_size,
        'stride' : h_stride,
        })

def max_pool(h_kernel_size, h_stride):
    def cfn(di, dh):
        def fn(di, isTraining=True):
            with tf.device('/gpu:0'):
                return {'Out' : tf.nn.max_pool(di['In'],
                    [1, dh['kernel_size'], dh['kernel_size'], 1], [1, dh['stride'], dh['stride'], 1], 'SAME')}
        return fn
    return siso_tfem('MaxPool2D', cfn, {
        'kernel_size' : h_kernel_size, 'stride' : h_stride,})

def keras_batch_normalization(name='default', weight_sharer=None):
    name = name + '_bn'
    def cfn(di, dh):
        bn = weight_sharer.get(name, tf.keras.layers.BatchNormalization,
            lambda layer: layer.get_weights())
        if not bn.built:
            with tf.device('/gpu:0'):
                bn.build(di['In'].get_shape())
                weights = weight_sharer.load_weights(name)
                if weights is not None:
                    bn.set_weights(weights)
        def fn(di, isTraining):
            with tf.device('/gpu:0'):
                return {'Out' : bn(di['In'], training=isTraining) }
        return fn
    return siso_tfem('BatchNormalization', cfn, {})

def relu():
    def cfn(di, dh):
        def fn(di, isTraining=True):
            with tf.device('/gpu:0'):
                return {'Out' : tf.nn.relu(di['In'])}
        return fn
    return siso_tfem('ReLU', cfn, {})

def conv2D(filter_size, name, weight_sharer, out_filters=None):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        channels = channels if out_filters is None else out_filters

        conv_fn = lambda: tf.keras.layers.Conv2D(channels, filter_size, padding='same')
        conv = weight_sharer.get(name + '_conv_' + str(filter_size), conv_fn, 
            lambda layer: layer.get_weights())
        if not conv.built:
            with tf.device('/gpu:0'):
                conv.build(di['In'].get_shape())
                weights = weight_sharer.load_weights(name)
                if weights is not None:
                    conv.set_weights(weights)
        def fn(di, isTraining=True):
            with tf.device('/gpu:0'):
                return {'Out' : conv(di['In'])}
        return fn

    return siso_tfem('Conv2D', cfn, {})

def conv2D_depth_separable(filter_size, name, weight_sharer, out_filters=None):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        channels = channels if out_filters is None else out_filters
        conv_fn = lambda: tf.keras.layers.SeparableConv2D(channels, filter_size, padding='same')
        conv = weight_sharer.get(name + '_dsep_' + str(filter_size), conv_fn, 
            lambda layer: layer.get_weights())
        if not conv.built:
            with tf.device('/gpu:0'):
                conv.build(di['In'].get_shape())
                weights = weight_sharer.load_weights(name)
                if weights is not None:
                    conv.set_weights(weights)
        
        def fn(di, isTraining=True):
            with tf.device('/gpu:0'):
                return {'Out' : conv(di['In'])}
        return fn

    return siso_tfem('Conv2DSeparable', cfn, {})

def global_pool():
    def cfn(di, dh):
        def fn(di, isTraining):
            with tf.device('/gpu:0'):
                return {'Out': tf.reduce_mean(di['In'], [1, 2])}
        return fn
    return TFEM('GlobalPool', {}, cfn, ['In'], ['Out']).get_io()

def dropout(keep_prob):
    def cfn(di, dh):
        def fn(di, isTraining=True):
            if isTraining:
                with tf.device('/gpu:0'):
                    out = tf.nn.dropout(di['In'], keep_prob)
            else:
                out = di['In']
            return {'Out': out}
        return fn
    return TFEM('Dropout', {}, cfn, ['In'], ['Out']).get_io()

def fc_layer(num_classes, name, weight_sharer):
    name = name + '_fc_layer_' + str(num_classes)
    def cfn(di, dh):
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
                return {'Out' : fc(di['In'])}
        return fn
    return siso_tfem('fc_layer', cfn, {}) 

def wrap_relu_batch_norm(io_pair, add_relu=True, add_bn=True, weight_sharer=None, name=None):
    assert add_relu or add_bn
    elements = [True, add_relu, add_bn]
    module_fns = [
        lambda: io_pair, 
        relu,
        lambda: keras_batch_normalization(name=name, weight_sharer=weight_sharer)]
    return mo.siso_sequential([module_fn() for i, module_fn in enumerate(module_fns) if elements[i]])

def wrap_batch_norm_relu(io_pair, add_relu=True, add_bn=True, weight_sharer=None, name=None):
    assert add_relu or add_bn
    elements = [True, add_bn, add_relu]
    module_fns = [
        lambda: io_pair, 
        lambda: keras_batch_normalization(name=name, weight_sharer=weight_sharer), 
        relu]
    return mo.siso_sequential([module_fn() for i, module_fn in enumerate(module_fns) if elements[i]])
