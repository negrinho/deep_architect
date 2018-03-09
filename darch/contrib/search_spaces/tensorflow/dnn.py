import darch.core as co
import darch.hyperparameters as hp
import darch.helpers.tensorflow as htf
import darch.modules as mo
import tensorflow as tf
import numpy as np

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

# auxiliary definitions
D = hp.Discrete

def siso_tfm(name, compile_fn, name_to_h={}, scope=None):
    return htf.TFModule(name, name_to_h, compile_fn, ['In'], ['Out'], scope).get_io()

# modules
def relu():
    return siso_tfm('ReLU', lambda di, dh: lambda di: {'Out' : tf.nn.relu(di['In'])})

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
        def fn(In):
            return {'Out' : tf.nn.dropout(di['In'], p)}
        return fn, {p : dh['keep_prob']}, {p : 1.0}
    return siso_tfm('Dropout', cfn, {'keep_prob' : h_keep_prob})

def batch_normalization():
    def cfn(di, dh):
        p_var = tf.placeholder(tf.bool)
        def fn(di):
            return {'Out' : tf.layers.batch_normalization(di['In'], training=p_var)}
        return fn, {p_var : 1}, {p_var : 0}
    return siso_tfm('BatchNormalization', cfn)

def conv2d(h_num_filters, h_filter_width, h_stride, h_W_init_fn, h_b_init_fn):
    def cfn(In, num_filters, filter_width, stride, W_init_fn, b_init_fn):
        (_, _, _, num_channels) = In.get_shape().as_list()
        W = tf.Variable(W_init_fn([filter_width, filter_width, num_channels, num_filters]))
        b = tf.Variable(b_init_fn([num_filters]))
        def fn(In):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(In, W, [1, stride, stride, 1], 'SAME'), b)}
        return fn
    return siso_tfm('Conv2D', cfn, {
        'num_filters' : h_num_filters,
        'filter_width' : h_filter_width,
        'stride' : h_stride,
        'W_init_fn' : h_W_init_fn,
        'b_init_fn' : h_b_init_fn,
        })

def conv2d_simplified(h_num_filters, h_filter_width, stride):
    W_init_fn = kaiming2015delving_initializer_conv()
    b_init_fn = constant_initializer(0.0)
    return conv2d(h_num_filters, h_filter_width,
        D([stride]), D([W_init_fn]), D([b_init_fn]))

def max_pool2d(h_kernel_size, h_stride):
    def cfn(kernel_size, stride):
        def fn(In):
            return {'Out' : tf.nn.max_pool(In,
                [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], 'SAME')}
        return fn
    return siso_tfm('MaxPool2D', cfn, {
        'kernel_size' : h_kernel_size, 'stride' : h_stride,})

def affine_simplified(h_m):
    def cfn(In, m):
        shape = In.get_shape().as_list()
        n = np.product(shape[1:])
        def fn(In):
            if len(shape) > 2:
                In = tf.reshape(In, [-1, n])
            return {'Out' : tf.layers.dense(In, m)}
        return fn
    return siso_tfm('AffineSimplified', cfn, {'m' : h_m})

def nonlinearity(h_or):
    def cfn(nonlin_name):
        def fn(In):
            if nonlin_name == 'relu':
                Out = tf.nn.relu(In)
            elif nonlin_name == 'relu6':
                Out = tf.nn.relu6(In)
            elif nonlin_name == 'crelu':
                Out = tf.nn.crelu(In)
            elif nonlin_name == 'elu':
                Out = tf.nn.elu(In)
            elif nonlin_name == 'softplus':
                Out = tf.nn.softplus(In)
            else:
                raise ValueError
            return {"Out" : Out}
        return fn
    return siso_tfm('Nonlinearity', cfn, {'idx' : h_or})

def dnn_cell(h_num_hidden, h_nonlin, h_swap,
        h_opt_drop, h_opt_bn, h_drop_keep_prob):
    return mo.siso_sequential([
        affine_simplified(h_num_hidden),
        nonlinearity(h_nonlin),
        mo.siso_permutation([
            lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),
            lambda: mo.siso_optional(lambda: batch_normalization(), h_opt_bn),
        ], h_swap)])