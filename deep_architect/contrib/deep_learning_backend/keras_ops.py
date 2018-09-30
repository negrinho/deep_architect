from keras import layers
from deep_architect.helpers.keras import siso_keras_module
from deep_architect.hyperparameters import Discrete as D

"""
di['In'] is expected to be a tuple representing the shape of a single input
"""
def input_node():
    def compile_fn(di, dh):
        def fn(di):
            return {'Out': layers.Input(di['In'])}
        return fn
    return siso_keras_module('Input', compile_fn, {})

def conv2d(h_num_filters, h_filter_width, h_stride, h_use_bias):
    def compile_fn(di, dh):
        layer = layers.Conv2D(dh['num_filters'], (dh['filter_width'],) * 2,
            strides=(dh['stride'],) * 2, use_bias=dh['use_bias'], padding='SAME')
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_keras_module('Conv2D', compile_fn, {
        'num_filters' : h_num_filters,
        'filter_width' : h_filter_width,
        'stride' : h_stride,
        'use_bias' : h_use_bias,
        })

def avg_pool2d(h_kernel_size, h_stride):
    def compile_fn(di, dh):
        layer = layers.AveragePooling2D(
            pool_size=dh['kernel_size'], 
            strides=(dh['stride'], dh['stride']), 
            padding='same')
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_keras_module('AvgPool', compile_fn, {
        'kernel_size' : h_kernel_size,
        'stride' : h_stride,
        })

def max_pool2d(h_kernel_size, h_stride):
    def compile_fn(di, dh):
        layer = layers.MaxPooling2D(
            pool_size=dh['kernel_size'], 
            strides=(dh['stride'], dh['stride']), 
            padding='same')
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_keras_module('MaxPool2D', compile_fn, {
        'kernel_size' : h_kernel_size, 
        'stride' : h_stride,
        })

def dropout(h_keep_prob):
    def compile_fn(di, dh):
        layer = layers.Dropout(dh['keep_prob'])
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_keras_module('Dropout', compile_fn, {'keep_prob' : h_keep_prob})

def batch_normalization():
    def compile_fn(di, dh):
        layer = layers.BatchNormalization()
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_keras_module('BatchNormalization', compile_fn, {})

def activation(h_activation):
    def compile_fn(di, dh):
        layer = layers.Activation(dh['activation'])
        def fn(di):
            return {'Out': layer(di['In'])}
        return fn
    return siso_keras_module('Activation', compile_fn, {'activation': h_activation})

def relu():
    return activation(D(['relu']))

def global_pool2d():
    def compile_fn(di, dh):
        layer = layers.GlobalAveragePooling2D()
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_keras_module('GlobalAveragePool', compile_fn, {})

def fc_layer(h_num_units):
    def compile_fn(di, dh):
        layer = layers.Dense(dh['num_units'])
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_keras_module('FCLayer', compile_fn, {'num_units' : h_num_units})

func_dict = {
    'dropout': dropout,
    'conv2d': conv2d,
    'max_pool2d': max_pool2d,
    'batch_normalization': batch_normalization,
    'relu': relu,
    'global_pool2d': global_pool2d,
    'fc_layer': fc_layer
}
