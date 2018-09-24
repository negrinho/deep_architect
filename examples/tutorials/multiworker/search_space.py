# Using TensorFlow Backend

# Search Space
import keras
import numpy as np

import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
from deep_architect.contrib.misc.search_spaces.tensorflow.common import siso_tfm

D = hp.Discrete # Discrete Hyperparameter

def flatten():
    def cfn(di, dh):
        Flatten = keras.layers.Flatten()
        def fn(di):
            return {'Out': Flatten(di['In'])}
        return fn
    return siso_tfm('Flatten', cfn, {}) # use siso_tfm for now

def dense(h_units):
    def cfn(di, dh):
        Dense = keras.layers.Dense(dh['units'])
        def fn(di):
            return {'Out' : Dense(di['In'])}
        return fn
    return siso_tfm('Dense', cfn, {'units' : h_units})

def nonlinearity(h_nonlin_name):
    def cfn(di, dh):
        def fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = keras.layers.Activation('relu')(di['In'])
            elif nonlin_name == 'tanh':
                Out = keras.layers.Activation('tanh')(di['In'])
            elif nonlin_name == 'elu':
                Out = keras.layers.Activation('elu')(di['In'])
            else:
                raise ValueError
            return {"Out" : Out}
        return fn
    return siso_tfm('Nonlinearity', cfn, {'nonlin_name' : h_nonlin_name})

def dropout(h_keep_prob):
    def cfn(di, dh):
        Dropout = keras.layers.Dropout(dh['keep_prob'])
        def fn(di):
            return {'Out' : Dropout(di['In'])}
        return fn
    return siso_tfm('Dropout', cfn, {'keep_prob' : h_keep_prob})

def batch_normalization():
    def cfn(di, dh):
        bn = keras.layers.BatchNormalization()
        def fn(di):
            return {'Out' : bn(di['In'])}
        return fn
    return siso_tfm('BatchNormalization', cfn, {})

def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn, h_drop_keep_prob):
    return mo.siso_sequential([
        dense(h_num_hidden),
        nonlinearity(h_nonlin_name),
        mo.siso_permutation([
            lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),
            lambda: mo.siso_optional(batch_normalization, h_opt_bn),
        ], h_swap)])

def dnn_net(num_classes):
    h_nonlin_name = D(['relu', 'tanh', 'elu'])
    h_swap = D([0, 1])
    h_opt_drop = D([0, 1])
    h_opt_bn = D([0, 1])
    return mo.siso_sequential([
        flatten(),
        mo.siso_repeat(lambda: dnn_cell(
            D([64, 128, 256, 512, 1024]),
            h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
            D([0.25, 0.5, 0.75])), D([1, 2])),
        dense(D([num_classes]))])