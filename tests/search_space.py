import tensorflow as tf
import darch.hyperparameters as hp
import darch.modules as mo
import darch.core as co
import darch.helpers.tensorflow as tf_helpers
import darch.utils as ut
import numpy as np


TFM = tf_helpers.TFModule
D = hp.Discrete

def SISOTFM(name, compile_fn, name_to_h={}, scope=None):
    return TFM(name, name_to_h, compile_fn, ['In'], ['Out'], scope)

def get_init_fn(s):
    x, y = s.split('=')
    if x == 'xavier':
        return xavier_initializer_affine( float(y) )
    elif x == 'const':
        return const_fn( float(y) ) 

trunc_normal_fn = lambda stddev: lambda shape: tf.truncated_normal(shape, stddev=stddev)
const_fn = lambda c: lambda shape: tf.constant(c, shape=shape)

def kaiming2015delving_initializer_conv(gain=1.0):
    def init_fn(shape):
        n = np.product(shape)
        stddev = gain * np.sqrt( 2.0 / n )
        init_vals = tf.random_normal(shape, 0.0, stddev)
        return init_vals
    return init_fn

def xavier_initializer_affine(gain=1.0):
    def init_fn(shape):
        n, m = shape
        
        sc = gain * ( np.sqrt(6.0) / np.sqrt(m + n) )
        init_vals = tf.random_uniform([n, m], -sc, sc)
        return init_vals
    return init_fn

# TODO: I need to encapsulate things.
# this starts here.
def affine_cfn(In, m, W_init, b_init):
    shape = In.get_shape().as_list()
    n = np.product(shape[1:])
    W = tf.Variable( get_init_fn(W_init)( [n, m] ) )
    b = tf.Variable( get_init_fn(b_init)( [m] ) )
    def fn(In):
        if len(shape) > 2:
            In = tf.reshape(In, [-1, n])
        Out = tf.add(tf.matmul(In, W), b)
        # print In.get_shape().as_list()
        return {'Out' : Out}
    return fn

def dropout_cfn(keep_prob):
    p = tf.placeholder(tf.float32)
    def fn(In):
        return {'Out' : tf.nn.dropout(In, p)} 
    return fn, {p : keep_prob}, {p : 1.0} 

def relu_cfn():
    def fn(In):
        return {'Out' : tf.nn.relu(In)}
    return fn

def batch_norm_cfn():
    p_var = tf.placeholder(tf.bool)
    def fn(In):
        return {'Out' : tf.layers.batch_normalization(In, training=p_var) }
    return fn, {p_var : 1}, {p_var : 0}     

# search modules.
def BatchNormalization():
    return SISOTFM('BatchNormalization', batch_norm_cfn)

def Dropout(h_keep_prob):
    return SISOTFM('Dropout', dropout_cfn, {'keep_prob' : h_keep_prob})

def ReLU():
    return SISOTFM('ReLU', relu_cfn)

def Affine(h_m, h_W_init, h_b_init):
    return SISOTFM('Affine', affine_cfn, 
        {'m' : h_m, 'W_init' : h_W_init, 'b_init' : h_b_init})

# TODO: some benchmarking for the searchers.
io_fn = lambda m: (m.inputs, m.outputs)
io_lst_fn = lambda m_lst: (m_lst[0].inputs, m_lst[-1].outputs)

def DNNCell(h_num_hidden, h_W_init, h_b_init, h_swap, 
        h_opt_drop, h_opt_bn, h_drop_keep_prob):
    ms = [
        Affine(h_num_hidden, h_W_init, h_b_init),
        ReLU(),
        mo.SISOPermutation([
            lambda: io_fn( 
                mo.SISOOptional(
                    lambda: io_fn( Dropout(h_drop_keep_prob) ), 
                    h_opt_drop ) ),
            lambda: io_fn( 
                mo.SISOOptional(
                    lambda: io_fn( BatchNormalization() ), 
                    h_opt_bn ) ),
        ], h_swap),
        mo.Empty()
    ]
    ut.connect_sequentially(ms)
    return ms

def Conv2D(h_num_filters, h_filter_size, h_stride, h_W_init, h_b_init):
    def cfn(In, num_filters, filter_size, stride, W_init, b_init):
        (_, height, width, channels) = In.get_shape().as_list()

        W = tf.Variable( get_init_fn(W_init)( [filter_size, filter_size, channels, num_filters] ) )
        b = tf.Variable( get_init_fn(b_init)( [num_filters] ) )
        def fn(In):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(In, W, [1, stride, stride, 1], 'SAME'), b)}
        return fn

    return SISOTFM('Conv2D', cfn, {
        'num_filters' : h_num_filters, 
        'filter_size' : h_filter_size, 
        'stride' : h_stride,
        'W_init' : h_W_init, 
        'b_init' : h_b_init,
        })

def MaxPool(h_kernel_size, h_stride):
    def cfn(kernel_size, stride):
        def fn(In):
            return {'Out' : tf.nn.max_pool(In, 
                [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], 'SAME')}
        return fn
    return SISOTFM('MaxPool', cfn, {
        'kernel_size' : h_kernel_size, 
        'stride' : h_stride,
        })

def Add():
    return tf_helpers.TFModule('Add', {}, 
        lambda: lambda In0, In1: tf.add(In0, In1), 
        ['In0', 'In1'], ['Out'])

def AffineSimplified(h_m):
    def cfn(In, m):
        shape = In.get_shape().as_list()
        n = np.product(shape[1:])

        def fn(In):
            if len(shape) > 2:
                In = tf.reshape(In, [-1, n])
            return {'Out' : tf.layers.dense(In, m)}
        return fn
    return SISOTFM('AffineSimplified', cfn, {'m' : h_m})

def Nonlinearity(h_or):
    def cfn(idx):
        def fn(In):
            if idx == 0:
                Out = tf.nn.relu(In)
            elif idx == 1:
                Out = tf.nn.relu6(In)
            elif idx == 2:
                Out = tf.nn.crelu(In)
            elif idx == 3:
                Out = tf.nn.elu(In)
            elif idx == 4:
                Out = tf.nn.softplus(In)
            else:
                raise ValueError
            return {"Out" : Out}
        return fn
    return SISOTFM('Nonlinearity', cfn, {'idx' : h_or})

# TODO: this is going to be important.
def search_space_fn():
    co.Scope.reset_default_scope()
    
    h_num_hidden = D([ 32, 64, 128, 256, 512, 1024 ])
    h_drop_keep_prob = D([ 0.3, 0.5, 0.9 ])
    h_W_init = D([ 'xavier=0.01', 'xavier=0.1', 'xavier=1.0' ])
    h_b_init = D([ 'const=0.0', 'const=0.05', 'const=0.1' ])

    h_swap = D([ 0, 1 ])
    h_opt_drop = D([ 0, 1 ])
    h_opt_bn = D([ 0, 1 ])
    h_reps = D([ 2, 4, 8, 16 ])

    # NOTE: all these cells are sharing the hyperparameters. this needs to be 
    # changed later.

    ms = [
        mo.Empty(),
        mo.SISORepeat(
            lambda: io_lst_fn(
                DNNCell(h_num_hidden, h_W_init, h_b_init, h_swap, 
                    h_opt_drop, h_opt_bn, h_drop_keep_prob) ),
            h_reps
        ),
        Affine( D([ 10 ]), h_W_init, h_b_init),
        mo.Empty()
    ]
    ut.connect_sequentially(ms)

    hs = {
        'optimizer_type' : D([ 'adam', 'sgd_mom' ]),
        'learning_rate_init' : D( np.logspace(-1, -5, num=16) ),
        'rate_mult' : D( np.logspace(-2, np.log10(0.9), num=8) ),
        'rate_patience' : D( range(8, 65, 4) ), 
        'stop_patience' : D([ 128 ]), 
        'learning_rate_min' : D([ 1e-7 ]),
        'angle_delta' : D([ 0, 5, 10, 15, 20, 25, 30, 35 ]),
        'scale_delta' : D([ 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35 ]),
        'weight_decay_coeff' : D([ 0.0, 1e-6, 1e-5, 1e-4 ]),
        }

    # hs = OrderedDict([
    #     ('optimizer_type', D([ 'adam', 'sgd_mom' ])),
    #     ('learning_rate_init', D( np.logspace(-1, -5, num=16) )),
    #     ('rate_mult', D( np.logspace(-2, np.log10(0.9), num=8) )),
    #     ('rate_patience', D( range(8, 65, 4) )), 
    #     ('stop_patience', D([ 128 ])), 
    #     ('learning_rate_min', D([ 1e-7 ])),
    #     ('angle_delta', D([ 0, 5, 10, 15, 20, 25, 30, 35 ])),
    #     ('scale_delta', D([ 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35 ])),
    #     ('weight_decay_coeff', D([ 0.0, 1e-6, 1e-5, 1e-4 ])),
    #     ])

    return ms[0].inputs, ms[-1].outputs, hs