from builtins import str
from keras import layers
import darch.modules as mo
from deep_architect.helpers import keras as hk
from deep_architect.hyperparameters import Discrete as D
KM = hk.KerasModule

def siso_km(name, compile_fn, name_to_hyperp, scope=None):
    return KM(name, name_to_hyperp, compile_fn, ['In'], ['Out'], scope).get_io()

"""
di['In'] is expected to be a tuple representing the shape of a single input
"""
def input_node():
    def cfn(di, dh):
        def fn(di):
            return {'Out': layers.Input(di['In'])}
        return fn
    return siso_km('Input', cfn, {})

def conv2d(h_num_filters, h_filter_width, h_stride, h_use_bias):
    def cfn(di, dh):
        layer = layers.Conv2D(dh['num_filters'], (dh['filter_width'],) * 2,
            strides=(dh['stride'],) * 2, use_bias=dh['use_bias'], padding='SAME')
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_km('Conv2D', cfn, {
        'num_filters' : h_num_filters,
        'filter_width' : h_filter_width,
        'stride' : h_stride,
        'use_bias' : h_use_bias,
        })

def avg_pool2d(h_kernel_size, h_stride):
    def cfn(di, dh):
        layer = layers.AveragePooling2D(
            pool_size=dh['kernel_size'], 
            strides=(dh['stride'], dh['stride']), 
            padding='same')
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_km('AvgPool', cfn, {
        'kernel_size' : h_kernel_size,
        'stride' : h_stride,
        })

def max_pool2d(h_kernel_size, h_stride):
    def cfn(di, dh):
        layer = layers.MaxPooling2D(
            pool_size=dh['kernel_size'], 
            strides=(dh['stride'], dh['stride']), 
            padding='same')
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_km('MaxPool2D', cfn, {
        'kernel_size' : h_kernel_size, 
        'stride' : h_stride,
        })

def dropout(h_keep_prob):
    def cfn(di, dh):
        layer = layers.Dropout(dh['keep_prob'])
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_km('Dropout', cfn, {'keep_prob' : h_keep_prob})

def batch_normalization():
    def cfn(di, dh):
        layer = layers.BatchNormalization()
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_km('BatchNormalization', cfn, {})

def activation(h_activation):
    def cfn(di, dh):
        layer = layers.Activation(dh['activation'])
        def fn(di):
            return {'Out': layer(di['In'])}
        return fn
    return siso_km('Activation', cfn, {'activation': h_activation})

def relu():
    return activation(D(['relu']))

def global_pool2d():
    def cfn(di, dh):
        layer = layers.GlobalAveragePooling2D()
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_km('GlobalAveragePool', cfn, {})

def fc_layer(h_num_units):
    def cfn(di, dh):
        layer = layers.Dense(dh['num_units'])
        def fn(di):
            return {'Out' : layer(di['In'])}
        return fn
    return siso_km('FCLayer', cfn, {'num_units' : h_num_units})

func_dict = {
    'dropout': dropout,
    'conv2d': conv2d,
    'max_pool2d': max_pool2d,
    'batch_normalization': batch_normalization,
    'relu': relu,
    'global_pool2d': global_pool2d,
    'fc_layer': fc_layer
}
