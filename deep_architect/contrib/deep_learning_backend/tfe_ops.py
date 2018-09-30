from deep_architect.helpers.tfeager import siso_tfeager_module
import tensorflow as tf

def max_pool2d(h_kernel_size, h_stride):
    def compile_fn(di, dh):
        def fn(di, isTraining=True):
            return {'Out' : tf.nn.max_pool(di['In'],
                [1, dh['kernel_size'], dh['kernel_size'], 1], [1, dh['stride'], dh['stride'], 1], 'SAME')}
        return fn
    return siso_tfeager_module('MaxPool2D', compile_fn, {
        'kernel_size' : h_kernel_size, 
        'stride' : h_stride,})

def batch_normalization():
    def compile_fn(di, dh):
        bn = tf.keras.layers.BatchNormalization()
        def fn(di, isTraining):
            return {'Out' : bn(di['In'], training=isTraining) }
        return fn
    return siso_tfeager_module('BatchNormalization', compile_fn, {})

def relu():
    def compile_fn(di, dh):
        def fn(di, isTraining=True):
            return {'Out' : tf.nn.relu(di['In'])}
        return fn
    return siso_tfeager_module('ReLU', compile_fn, {})

def conv2d(h_num_filters, h_filter_width, h_stride, h_use_bias):
    def compile_fn(di, dh):
        conv = tf.keras.layers.Conv2D(dh['num_filters'], dh['filter_width'],
            dh['stride'], use_bias=dh['use_bias'], padding='SAME')
        def fn(di, isTraining=True):
            return {'Out' : conv(di['In'])}
        return fn
    return siso_tfeager_module('Conv2D', compile_fn, {
        'num_filters' : h_num_filters,
        'filter_width' : h_filter_width,
        'stride' : h_stride,
        'use_bias' : h_use_bias
        })

def dropout(h_keep_prob):
    def compile_fn(di, dh):
        def fn(di, isTraining=True):
            if isTraining:
                out = tf.nn.dropout(di['In'], dh['keep_prob'])
            else:
                out = di['In']
            return {'Out': out}
        return fn
    return siso_tfeager_module('Dropout', compile_fn, {'keep_prob' : h_keep_prob})

def global_pool2d():
    def compile_fn(di, dh):
        def fn(di, isTraining=True):
            return {'Out' : tf.reduce_mean(di['In'], [1,2])}
        return fn
    return siso_tfeager_module('GlobalAveragePool', compile_fn, {})

def fc_layer(h_num_units):
    def compile_fn(di, dh):
        fc = tf.keras.layers.Dense(dh['num_units'])
        def fn(di, isTraining=True):
            return {'Out' : fc(di['In'])}
        return fn
    return siso_tfeager_module('FCLayer', compile_fn, {'num_units' : h_num_units})

func_dict = {
    'dropout': dropout,
    'conv2d': conv2d,
    'max_pool2d': max_pool2d,
    'batch_normalization': batch_normalization,
    'relu': relu,
    'global_pool2d': global_pool2d,
    'fc_layer': fc_layer
}