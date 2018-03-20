
import darch.core as co
import darch.modules as mo
import tensorflow as tf
import numpy as np
from darch.contrib.search_spaces.tensorflow.common import siso_tfm, D


# initializers
def constant_initializer(c):
    def init_fn(shape):
        return tf.constant(c, shape=shape)
    return init_fn

def truncated_normal_initializer(stddev):
    def init_fn(shape):
        return tf.truncated_normal(shape, stddev=stddev)
    return init_fn

def kaiming2015delving_initializer_conv(gain=1.0):
    def init_fn(shape):
        n = np.product(shape)
        stddev = gain * np.sqrt(2.0 / n)
        init_vals = tf.random_normal(shape, 0.0, stddev)
        return init_vals
    return init_fn

def xavier_initializer_affine(gain=1.0):
    def init_fn(shape):
        print shape
        n, m = shape

        sc = gain * (np.sqrt(6.0) / np.sqrt(m + n))
        init_vals = tf.random_uniform([n, m], -sc, sc)
        return init_vals
    return init_fn

# modules
def relu():
    return siso_tfm('ReLU', lambda di, dh: lambda di: {'Out' : tf.nn.relu(di['In'])}, {})

def affine(h_m, h_W_init_fn, h_b_init_fn):
    def cfn(di, dh):
        m = dh['m']
        shape = di['In'].get_shape().as_list()
        n = np.product(shape[1:])
        W = tf.Variable(dh['W_init_fn']([n, m]))
        b = tf.Variable(dh['b_init_fn']([m]))
        def fn(di):
            In = di['In']
            if len(shape) > 2:
                In = tf.reshape(In, [-1, n])
            return {'Out' : tf.add(tf.matmul(In, W), b)}
        return fn
    return siso_tfm('Affine', cfn,
        {'m' : h_m, 'W_init_fn' : h_W_init_fn, 'b_init_fn' : h_b_init_fn})

def dropout(h_keep_prob):
    def cfn(di, dh):
        p = tf.placeholder(tf.float32)
        def fn(di):
            return {'Out' : tf.nn.dropout(di['In'], p)}
        return fn, {p : dh['keep_prob']}, {p : 1.0}
    return siso_tfm('Dropout', cfn, {'keep_prob' : h_keep_prob})

def batch_normalization():
    def cfn(di, dh):
        p_var = tf.placeholder(tf.bool)
        def fn(di):
            return {'Out' : tf.layers.batch_normalization(di['In'], training=p_var)}
        return fn, {p_var : 1}, {p_var : 0}
    return siso_tfm('BatchNormalization', cfn, {})

# add having a bias or not.

def affine_simplified(h_m):
    def cfn(di, dh):
        shape = di['In'].get_shape().as_list()
        n = np.product(shape[1:])
        def fn(di):
            In = di['In']
            if len(shape) > 2:
                In = tf.reshape(In, [-1, n])
            return {'Out' : tf.layers.dense(In, dh['m'])}
        return fn
    return siso_tfm('AffineSimplified', cfn, {'m' : h_m})

def nonlinearity(h_nonlin_name):
    def cfn(di, dh):
        def fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = tf.nn.relu(di['In'])
            elif nonlin_name == 'relu6':
                Out = tf.nn.relu6(di['In'])
            elif nonlin_name == 'crelu':
                Out = tf.nn.crelu(di['In'])
            elif nonlin_name == 'elu':
                Out = tf.nn.elu(di['In'])
            elif nonlin_name == 'softplus':
                Out = tf.nn.softplus(di['In'])
            else:
                raise ValueError
            return {"Out" : Out}
        return fn
    return siso_tfm('Nonlinearity', cfn, {'nonlin_name' : h_nonlin_name})

def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn, h_drop_keep_prob):
    return mo.siso_sequential([
        affine_simplified(h_num_hidden),
        nonlinearity(h_nonlin_name),
        mo.siso_permutation([
            lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),
            lambda: mo.siso_optional(batch_normalization, h_opt_bn),
        ], h_swap)])

def dnn_net(num_classes):
    h_nonlin_name = D(['relu', 'relu6', 'crelu', 'elu', 'softplus'])
    h_swap = D([0, 1])
    h_opt_drop = D([0, 1])
    h_opt_bn = D([0, 1])
    return mo.siso_sequential([
        mo.siso_repeat(lambda: dnn_cell(
            D([64, 128, 256, 512, 1024]), 
            h_nonlin_name, h_swap, h_opt_drop, h_opt_bn, 
            D([0.25, 0.5, 0.75])), D([1, 2, 4, 8])),
        affine_simplified(D([num_classes]))])