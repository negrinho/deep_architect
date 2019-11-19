import deep_architect.modules as mo
import tensorflow as tf
import numpy as np
from deep_architect.contrib.misc.search_spaces.tensorflow.common import siso_tensorflow_module, D


# initializers
def constant_initializer(c):

    def init_fn(shape):
        return tf.constant(c, shape=shape)

    return init_fn


def truncated_normal_initializer(stddev):

    def init_fn(shape):
        return tf.truncated_normal(shape, stddev=stddev)

    return init_fn


def xavier_initializer_affine(gain=1.0):

    def init_fn(shape):
        n, m = shape

        sc = gain * (np.sqrt(6.0) / np.sqrt(m + n))
        init_vals = tf.random_uniform([n, m], -sc, sc)
        return init_vals

    return init_fn


# modules
def relu():
    return siso_tensorflow_module(
        'ReLU', lambda di, dh: lambda di: {'out': tf.nn.relu(di['in'])}, {})


def affine(h_num_hidden, h_W_init_fn, h_b_init_fn):

    def compile_fn(di, dh):
        m = dh['num_hidden']
        shape = di['in'].get_shape().as_list()
        n = np.product(shape[1:])
        W = tf.Variable(dh['W_init_fn']([n, m]))
        b = tf.Variable(dh['b_init_fn']([m]))

        def forward_fn(di):
            In = di['in']
            if len(shape) > 2:
                In = tf.reshape(In, [-1, n])
            return {'out': tf.add(tf.matmul(In, W), b)}

        return forward_fn

    return siso_tensorflow_module(
        'Affine', compile_fn, {
            'num_hidden': h_num_hidden,
            'W_init_fn': h_W_init_fn,
            'b_init_fn': h_b_init_fn
        })


def dropout(h_keep_prob):

    def compile_fn(di, dh):
        p = tf.placeholder(tf.float32)

        def forward_fn(di):
            return {'out': tf.nn.dropout(di['in'], p)}

        return forward_fn, {p: dh['keep_prob']}, {p: 1.0}

    return siso_tensorflow_module('Dropout', compile_fn,
                                  {'keep_prob': h_keep_prob})


def batch_normalization():

    def compile_fn(di, dh):
        p_var = tf.placeholder(tf.bool)

        def forward_fn(di):
            return {
                'out': tf.layers.batch_normalization(di['in'], training=p_var)
            }

        return forward_fn, {p_var: 1}, {p_var: 0}

    return siso_tensorflow_module('BatchNormalization', compile_fn, {})


def affine_simplified(h_num_hidden):

    def compile_fn(di, dh):
        shape = di['in'].get_shape().as_list()
        n = np.product(shape[1:])

        def forward_fn(di):
            In = di['in']
            if len(shape) > 2:
                In = tf.reshape(In, [-1, n])
            return {'out': tf.layers.dense(In, dh['num_hidden'])}

        return forward_fn

    return siso_tensorflow_module('AffineSimplified', compile_fn,
                                  {'num_hidden': h_num_hidden})


def nonlinearity(h_nonlin_name):

    def compile_fn(di, dh):

        def forward_fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = tf.nn.relu(di['in'])
            elif nonlin_name == 'relu6':
                Out = tf.nn.relu6(di['in'])
            elif nonlin_name == 'crelu':
                Out = tf.nn.crelu(di['in'])
            elif nonlin_name == 'elu':
                Out = tf.nn.elu(di['in'])
            elif nonlin_name == 'softplus':
                Out = tf.nn.softplus(di['in'])
            else:
                raise ValueError
            return {"out": Out}

        return forward_fn

    return siso_tensorflow_module('Nonlinearity', compile_fn,
                                  {'nonlin_name': h_nonlin_name})


def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
             h_drop_keep_prob):
    return mo.siso_sequential([
        affine_simplified(h_num_hidden),
        nonlinearity(h_nonlin_name),
        mo.siso_permutation([
            lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob),
                                     h_opt_drop),
            lambda: mo.siso_optional(batch_normalization, h_opt_bn),
        ], h_swap)
    ])


def dnn_net(num_classes):
    h_nonlin_name = D(['relu', 'relu6', 'crelu', 'elu', 'softplus'],
                      name='Mutatable')
    h_swap = D([0, 1], name='Mutatable_sub')
    h_opt_drop = D([0, 1], name='Mutatable_sub')
    h_opt_bn = D([0, 1], name='Mutatable_sub')
    return mo.siso_sequential([
        mo.siso_repeat(
            lambda: dnn_cell(D([64, 128, 256, 512, 1024]),
                             h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
                             D([0.25, 0.5, 0.75])), D([1, 2])),
        affine_simplified(D([num_classes]))
    ])
