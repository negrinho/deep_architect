
import darch.hyperparameters as hps
import darch.core as co
import darch.modules as mo
import darch.hyperparameters as hps
import darch.utils as ut
import darch.tensorflow_helpers as tf_helpers
import darch.searchers as se
import numpy as np
import tensorflow as tf

TFM = tf_helpers.TFModule
D = hps.Discrete

def SISOTFM(name, compile_fn, name_to_h={}, scope=None):
    return tf_helpers.TFModule(name, name_to_h, compile_fn, 
            ['In'], ['Out'], scope)

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

def relu_cfn():
    def fn(In):
        return {'Out' : tf.nn.relu(In)}
    return fn

def affine_cfn(In, m, W_init_fn, b_init_fn):
    shape = In.get_shape().as_list()
    n = np.product(shape[1:])
    W = tf.Variable( W_init_fn( [n, m] ) )
    b = tf.Variable( b_init_fn( [m] ) )
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

def batch_norm_cfn():
    p_var = tf.placeholder(tf.bool)
    def fn(In):
        return {'Out' : tf.layers.batch_normalization(In, training=p_var) }
    return fn, {p_var : 1}, {p_var : 0}     

def ReLU():
    return SISOTFM('ReLU', relu_cfn)

def Affine(h_m, h_W_init_fn, h_b_init_fn):
    return SISOTFM('Affine', affine_cfn, 
        {'m' : h_m, 'W_init_fn' : h_W_init_fn, 'b_init_fn' : h_b_init_fn})

def Dropout(h_keep_prob):
    return SISOTFM('Dropout', dropout_cfn, {'keep_prob' : h_keep_prob})
    
# TODO: perhaps add hyperparameters.
def BatchNormalization():
    return SISOTFM('BatchNormalization', batch_norm_cfn)

# takes a function, and creates this.
def Residual(main_fn, res_fn, combine_fn):
    (m_inputs, m_outputs) = main_fn()
    (r_inputs, r_outputs) = res_fn()
    (c_inputs, c_outputs) = combine_fn()

    i_inputs, i_outputs = io_fn( mo.Empty() )
    i_outputs['Out'].connect( m_inputs['In'] )
    i_outputs['Out'].connect( r_inputs['In'] )

    m_outputs['Out'].connect( c_inputs['In0'] )
    r_outputs['Out'].connect( c_inputs['In1'] )

    return (i_inputs, c_outputs)

def Add():
    return tf_helpers.TFModule('Add', {}, 
        lambda: lambda In0, In1: tf.add(In0, In1), 
        ['In0', 'In1'], ['Out'])

io_fn = lambda m: (m.inputs, m.outputs)
io_lst_fn = lambda m_lst: (m_lst[0].inputs, m_lst[-1].outputs)

def DNNCell(h_num_hidden, h_W_init_fn, h_b_init_fn, h_swap, 
        h_opt_drop, h_opt_bn, h_drop_keep_prob):
    ms = [
        Affine(h_num_hidden, h_W_init_fn, h_b_init_fn),
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

def get_hyperparameters():
    return {
        'optimizer_type' : D([ 'sgd_mom', 'adam' ]),
        'learning_rate_init' : D( np.logspace(-3, -5, num=32) ),
        'rate_mult' : D( np.logspace(-2, np.log10(0.9), num=8) ),
        'rate_patience' : D( range(4, 17, 2) ),
        'stop_patience' : D( range(4, 33, 2) ),
        'learning_rate_min' : D( [1e-7] ),
        # 'regularization_weight' : D( np.logspace(-4, -8, num=2) ),
        'regularization_weight' : D([ 1e-9 ]),
        'weight_decay_coeff' : D( np.logspace(-4, -9, num=8) ),
        # 'weight_decay_coeff' : D([ 1e-9 ])    
        }

if __name__ == '__main__':
    h_num_hidden = D([ 32, 64, 128, 256, 512, 1024 ])
    h_drop_keep_prob = D([ 0.3, 0.5, 0.9 ])
    h_W_init_fn = D([ xavier_initializer_affine(g) for g in [0.01, 0.1, 1.0] ])
    h_b_init_fn = D([ const_fn(c) for c in [0.0, 0.05, 0.1] ])
    h_swap = D([ 0, 1 ])
    h_opt_drop = D([ 0, 1 ])
    h_opt_bn = D([ 0, 1 ])

    fn = lambda: Residual( 
        lambda: io_lst_fn(
            DNNCell(h_num_hidden, h_W_init_fn, h_b_init_fn, h_swap, 
                h_opt_drop, h_opt_bn, h_drop_keep_prob) ),
        lambda: io_fn( mo.Empty() ),
        lambda: io_fn( Add() ) ) 

    inputs, outputs = Residual(
        lambda: io_fn( mo.SISORepeat(fn, D([ 2 ])) ),
        lambda: io_fn( mo.Empty() ),
        lambda: io_fn( Add() ) ) 

    fn_first = lambda: Residual(
        lambda: io_lst_fn(
            DNNCell(h_num_hidden, h_W_init_fn, h_b_init_fn, h_swap, 
                h_opt_drop, h_opt_bn, h_drop_keep_prob) ),        
        lambda: io_fn( mo.Empty() ),
        lambda: io_fn( Add() ) )

    res_fn = lambda: io_lst_fn(
            DNNCell(h_num_hidden, h_W_init_fn, h_b_init_fn, h_swap, 
                h_opt_drop, h_opt_bn, h_drop_keep_prob) )

    fn_iter = lambda inputs, outputs: Residual(
        lambda: (inputs, outputs),
        res_fn,
        lambda: io_fn( Add() ) )

    ms = [
        mo.SISONestedRepeat(fn_first, fn_iter, D([ 3 ])),
        mo.Empty()]
    
    ut.connect_sequentially(ms)
    se.random_specify(ms[-1:])
    ut.draw_graph(ms[-1:])

    print se.extract_features(ms[0].inputs, ms[-1].outputs, {})
    # co.forward({ inputs['In'] : tf.placeholder(tf.float32, [None, 64])})


    # ms = DNNCell(h_num_hidden, h_W_init_fn, h_b_init_fn, h_swap, 
    #     h_opt_drop, h_opt_bn, h_drop_keep_prob)

    # se.random_specify(ms[-1:])
    # ut.draw_graph(ms[-1:], True, True)

    # NOTE: this is going to be important. it only actually 
    # matters for the last one.
