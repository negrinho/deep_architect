import darch.core as co
import darch.hyperparameters as hp
import darch.utils as ut
import darch.helpers.tensorflow as htf
import darch.modules as mo
import tensorflow as tf
import numpy as np
import random

TFM = htf.TFModule
D = hp.Discrete

def SISOTFM(name, compile_fn, name_to_h={}, scope=None):
    return htf.TFModule(name, name_to_h, compile_fn, 
            ['In'], ['Out'], scope)

def TISOTFM(name, compile_fn, name_to_h={}, scope=None):
    return htf.TFModule(name, name_to_h, compile_fn, 
            ['In0', 'In1'], ['Out'], scope)

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
        print shape
        n, m = shape
        
        sc = gain * ( np.sqrt(6.0) / np.sqrt(m + n) )
        init_vals = tf.random_uniform([n, m], -sc, sc)
        return init_vals
    return init_fn

def ReLU():
    def cfn():
        def fn(In):
            return {'Out' : tf.nn.relu(In)}
        return fn
    return SISOTFM('ReLU', cfn)

def Affine(h_m, h_W_init_fn, h_b_init_fn):
    def cfn(In, m, W_init_fn, b_init_fn):
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
    return SISOTFM('Affine', cfn, 
        {'m' : h_m, 'W_init_fn' : h_W_init_fn, 'b_init_fn' : h_b_init_fn})

def Dropout(h_keep_prob):
    def cfn(keep_prob):
        p = tf.placeholder(tf.float32)
        def fn(In):
            return {'Out' : tf.nn.dropout(In, p)} 
        return fn, {p : keep_prob}, {p : 1.0} 
    return SISOTFM('Dropout', cfn, {'keep_prob' : h_keep_prob})
    
# TODO: perhaps add hyperparameters.
def BatchNormalization():
    def cfn():
        p_var = tf.placeholder(tf.bool)
        def fn(In):
            return {'Out' : tf.layers.batch_normalization(In, training=p_var) }
        return fn, {p_var : 1}, {p_var : 0}     
    return SISOTFM('BatchNormalization', cfn)

def Conv2D(h_num_filters, h_filter_size, h_stride, h_W_init_fn, h_b_init_fn):
    def cfn(In, num_filters, filter_size, stride, W_init_fn, b_init_fn):
        (_, height, width, channels) = In.get_shape().as_list()

        W = tf.Variable( W_init_fn( [filter_size, filter_size, channels, num_filters] ) )
        b = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(In):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(In, W, [1, stride, stride, 1], 'SAME'), b)}
        return fn

    return SISOTFM('Conv2D', cfn, {
        'num_filters' : h_num_filters, 
        'filter_size' : h_filter_size, 
        'stride' : h_stride,
        'W_init_fn' : h_W_init_fn, 
        'b_init_fn' : h_b_init_fn,
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

def AvgPool(h_kernel_size, h_stride):
    def cfn(kernel_size, stride):
        def fn(In):
            return {'Out' : tf.nn.avg_pool(In, 
                [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], 'SAME')}
        return fn
    return SISOTFM('AvgPool', cfn, {
        'kernel_size' : h_kernel_size, 
        'stride' : h_stride,
        })

def Add():
    return htf.TFModule('Add', {}, 
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

def DNNCell(h_num_hidden, h_nonlin, h_swap, 
        h_opt_drop, h_opt_bn, h_drop_keep_prob):
    ms = [
        AffineSimplified(h_num_hidden),
        Nonlinearity(h_nonlin),
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
    return io_lst_fn( ms )

io_fn = lambda m: (m.inputs, m.outputs)
io_lst_fn = lambda m_lst: (m_lst[0].inputs, m_lst[-1].outputs)

def io_fn2(in0, in1, out):
    return {'In0': in0, 'In1': in1}, {'Out': out}

def ResidualSimplified(fn):
    return mo.SISOResidual(fn, lambda: io_fn( mo.Empty() ), lambda: io_fn( Add() ))

def Conv2DSimplified(h_num_filters, h_filter_size, stride):
    def cfn(In, num_filters, filter_size):
        (_, _, _, channels) = In.get_shape().as_list()
        W_init_fn = kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        W = tf.Variable( W_init_fn( [filter_size, filter_size, channels, num_filters] ) )
        b = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(In):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(In, W, [1, stride, stride, 1], 'SAME'), b)}
        return fn

    return SISOTFM('Conv2DSimplified', cfn, {
        'num_filters' : h_num_filters,
        'filter_size' : h_filter_size})

def ConvBottleNeck(h_filter_size, h_stride):
    def cfn(In, filter_size, stride):
        (_, _, _, channels) = In.get_shape().as_list()
        W_init_fn = kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        if(stride == 2):
            num_filters = 2 * channels

        W1 = tf.Variable( W_init_fn( [1, filter_size, channels, num_filters] ) )
        b1 = tf.Variable( b_init_fn( [num_filters] ) )
        W2 = tf.Variable( W_init_fn( [filter_size, 1, num_filters, num_filters] ) )
        b2 = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(In):
            intermediate = tf.nn.bias_add(
                tf.nn.conv2d(In, W1, [1, stride, stride, 1], 'SAME'), b1)
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(intermediate, W2, [1, 1, 1, 1], 'SAME'), b2)}
        return fn

    return SISOTFM('Conv2DSimplified', cfn, {
        'filter_size' : h_filter_size,
        'stride' : h_stride})
        


def Conv2DDilated(h_filter_size, h_dilation, h_stride):
    def cfn(In, filter_size, dilation, stride):
        (_, _, _, channels) = In.get_shape().as_list()
        W_init_fn = kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        if(stride == 2):
            num_filters = 2 * channels

        W = tf.Variable( W_init_fn( [filter_size, filter_size, channels, num_filters] ) )
        b = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(In):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(In, W, [1, stride, stride, 1], 'SAME', dilations=[1, dilation, dilation, 1]), b)}
        return fn

    return SISOTFM('Conv2DDilated', cfn, {
        'filter_size' : h_filter_size,
        'dilation' : h_dilation,
        'stride' : h_stride})

def Conv2DSeparable(h_filter_size, h_channel_multiplier, h_stride):
    def cfn(In, filter_size, channel_multiplier, stride):
        (_, _, _, channels) = In.get_shape().as_list()
        W_init_fn = kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        if(stride == 2):
            num_filters = 2 * channels

        W_depth = tf.Variable( W_init_fn( [filter_size, filter_size, channels, channel_multiplier] ) )
        W_point = tf.Variable( W_init_fn( [1, 1, channels * channel_multiplier, num_filters] ) )
        b = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(In):
            return {'Out' : tf.nn.bias_add(
                tf.nn.separable_conv2d(In, W_depth, W_point, [1, stride, stride, 1], 'SAME', [1, 1]), b)}
        return fn

    return SISOTFM('Conv2DSeparable', cfn, {
        'filter_size' : h_filter_size,
        'channel_multiplier' : h_channel_multiplier,
        'stride' : h_stride})

def SP1Operation(h_op_name, h_stride):
    return SISOOr({
            'identity': lambda: io_fn(mo.Empty()),
            'sep3': lambda: io_fn(Conv2DSeparable(D([3]), D([1]), h_stride)),
            'sep5': lambda: io_fn(Conv2DSeparable(D([5]), D([1]), h_stride)),
            'sep7': lambda: io_fn(Conv2DSeparable(D([7]), D([1]), h_stride)),
            'avg3': lambda: io_fn(AvgPool(D([3]), h_stride)),
            'max3': lambda: io_fn(MaxPool(D([3]), h_stride)),
            'dil3': lambda: io_fn(Conv2DDilated(D([3]), D([2]), h_stride)),
            'bot7': lambda: io_fn(ConvBottleNeck(D([7]), h_stride))
            }, h_op_name)

def SP1Combine(h_op1_name, h_op2_name, h_stride):
    in1 = lambda: io_fn(SP1Operation(h_op1_name, h_stride))
    in2 = lambda: io_fn(SP1Operation(h_op2_name, h_stride))
    add = lambda: io_fn(Add())
    return mo.Combine([in1, in2], add) 

def NormalCellSP1(h_N):
    def combine():
        C = 5
        ins = mo.Empty(num_connections=2)
        available = [output['Out'] for output in ins.outputs]
        unused = [False] * (C + 2)
        for i in xrange(C):
            h_in1_pos = D(range(len(available)))
            h_in2_pos = D(range(len(available)))

            def select(selection):
                unused[selection] = True
                return available[selection]
            
            h_op1 = D(['identity', 'sep3', 'sep5', 'sep7', 'avg3', 'max3', 'dil3', 'bot7'])
            h_op2 = D(['identity', 'sep3', 'sep5', 'sep7', 'avg3', 'max3', 'dil3', 'bot7'])
            combination = SP1Combine(h_op1, h_op2, D([1]))
            inputs, outputs = io_fn(combination)
            mo.Selector(select, h_in1_pos).connect(inputs['In1'])
            available.append(combination)




    def connect_prev(N, h)

    def cfn(N):
        for i in range(N):
            h_op1 = D(['identity', 'sep3', 'sep5', 'sep7', 'avg3', 'max3', 'dil3', 'bot7'])
            h_op2 = D(['identity', 'sep3', 'sep5', 'sep7', 'avg3', 'max3', 'dil3', 'bot7'])
            h_in1 = D(range(N+1))
            h_in2 = D(range(N+1))
            newOut = SP1Combine(h_op1, h_op2, D([1]))


### my interpretation.
def hyperparameters_fn():
    return {
        'optimizer_type' : D([ 'adam' ]),
        'lr_start' : D( np.logspace(-1, -4, num=16) ),
        'stop_patience' : D([ 512 ]), 
        'lr_end' : D([ 1e-6 ]),
        # 'max_num_epochs' : D([ 100 ]),
        # 'angle_delta' : D([ 0, 5, 10, 15, 20, 25, 30, 35 ]),
        # 'scale_delta' : D([ 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35 ]),
        'weight_decay_coeff' : D([ 0.0, 1e-6, 1e-5, 1e-4 ]),
        # 'scale_delta' : D([ 0.0, 0.05, 0.1, 0.2 ]),
        # 'angle_delta' : D([ 0, 5, 10, 15, 20 ])
        }

def get_ss0_fn(num_classes):
    def fn():
        def cell_fn():
            h_num_hidden = D([ 64, 128, 256, 512, 1024])
            h_nonlin = D([ 0, 1, 2, 3, 4 ])
            h_swap = D([ 0, 1 ])
            h_opt_drop = D([ 0, 1 ])
            h_opt_bn = D([ 0, 1 ])
            h_drop_keep_prob = D([ 0.3, 0.5, 0.7 ])

            return DNNCell(h_num_hidden, h_nonlin, h_swap, 
                h_opt_drop, h_opt_bn, h_drop_keep_prob)

        co.Scope.reset_default_scope()
        ms = [
            mo.Empty(),
            mo.SISORepeat( lambda: io_fn( 
                mo.SISORepeat(cell_fn, 
                D([1, 2, 4]) ) ), 
            D([1, 2, 4]) ),
            AffineSimplified( D([ num_classes ]) )
        ]
        ut.connect_sequentially(ms)

        return ms[0].inputs, ms[-1].outputs, hyperparameters_fn()
    return fn

def get_ss1_fn(num_classes):
    def fn():
        co.Scope.reset_default_scope()

        h_m = D([ num_classes ])
        h_W_init = D([ xavier_initializer_affine() ])
        h_b_init = D([ const_fn(0.0) ])

        ms = [
            Affine(h_m, h_W_init, h_b_init)
        ]
        ut.connect_sequentially(ms)
        
        return ms[0].inputs, ms[-1].outputs, hyperparameters_fn()
    return fn

def get_sample_space(num_classes):
    def fn():
        co.Scope.reset_default_scope()
        h_N = D([4, 6])
        h_F = D([32, 64, 128])
        def create_normal_cell():
            def normal_cell(in_0, in_1):
                inputs_lst = 
        def create_overall(N, F):
            


    return fn

def get_ss2_fn(num_classes):
    def fn():
        co.Scope.reset_default_scope()

        def fn1(h_num_filters, h_filter_size, h_perm, h_opt):
            ms = [
                Conv2DSimplified(h_num_filters, h_filter_size, 1),
                mo.SISOPermutation([
                    lambda: io_fn( ReLU() ), 
                    lambda: io_fn( mo.SISOOptional(
                        lambda: io_fn( BatchNormalization() ), h_opt ) )
                ], h_perm),
            ]
            ut.connect_sequentially(ms)
            return ms[0].inputs, ms[-1].outputs

        def fn2(h_num_filters, h_filter_size, h_perm, h_opt, h_or):
            f = lambda: fn1(h_num_filters, h_filter_size, h_perm, h_opt)
            return io_fn( 
                mo.SISOOr([
                    f, f
                    # lambda: ResidualSimplified(f)  
                ], h_or) )

        def fn3(hr_num_filters, hr_filter_size, ho_num_filters, ho_filter_size, h_opt, h_or):
            ms = [
                mo.SISOOr([
                    lambda: io_fn( Conv2DSimplified(hr_num_filters, hr_filter_size, 2) ),
                    lambda: io_fn( MaxPool(hr_filter_size, D([ 2 ])) ) ], h_or),
                mo.SISOOptional( 
                    lambda: io_fn( Conv2DSimplified(ho_num_filters, ho_filter_size, 1) ), h_opt ),
            ]
            ut.connect_sequentially(ms)

            return io_lst_fn( ms )        

        h_f = D([ 32, 64, 128 ])
        h_rep = D([ 1, 2, 4, 8 ])
        h_perm = hp.Bool()
        h_or = hp.Bool()
        h_opt = hp.Bool()
        fs = D([ 3 ])

        f1 = lambda: fn2(h_f, fs, h_perm, h_opt, h_or)
        f2 = lambda: fn2(h_f, fs, h_perm, h_opt, h_or)
        f3 = lambda: fn2(h_f, fs, h_perm, h_opt, h_or)

        h3_or = hp.Bool()
        h3_opt = hp.Bool()

        lst = [
            io_fn( mo.Empty() ),
            fn3(h_f, fs, h_f, fs, h3_opt, h3_or),
            io_fn( mo.SISORepeat(f1, h_rep) ),
            fn3(h_f, fs, h_f, fs, h3_opt, h3_or),
            io_fn( mo.SISORepeat(f2, h_rep) ),
            fn3(h_f, fs, h_f, fs, h3_opt, h3_or),
            io_fn( mo.SISORepeat(f3, h_rep) ),
            fn3(h_f, fs, h_f, fs, h3_opt, h3_or),
            io_fn( AffineSimplified(D([ 2 ])) )
        ]

        for (ins_prev, outs_prev), (ins_next, outs_next) in zip(lst[:-1], lst[1:]):
            outs_prev['Out'].connect( ins_next['In'] )

        return lst[0][0], lst[-1][1], hyperparameters_fn()
    return fn

# TODO: Fix the dependent hyperparameters.

###
# import tensorflow.contrib.slim as slim

### TODO: it is possible 

# def resUnit(input_layer,i):
#     with tf.variable_scope("res_unit" + str(i)):
#         part1 = slim.batch_norm(input_layer, activation_fn=None)
#         part2 = tf.nn.relu(part1)
#         part3 = slim.conv2d(part2, 64, [3, 3], activation_fn=None)
#         part4 = slim.batch_norm(part3, activation_fn=None)
#         part5 = tf.nn.relu(part4)
#         part6 = slim.conv2d(part5, 64, [3, 3], activation_fn=None)
#         output = input_layer + part6
#         return output

# def get_search_space_fn(num_classes):
#     def ResNet():
#         co.Scope.reset_default_scope()    
        
#         total_layers = 25
#         units_between_stride = total_layers / 5
        
#         def fn(In):
#             layer1 = slim.conv2d(In, 64, [3, 3], 
#                 normalizer_fn=slim.batch_norm, scope='conv_' + str(0))
#             for i in range(5):
#                 for j in range(units_between_stride):
#                     layer1 = resUnit(layer1, j + (i*units_between_stride))
#                 layer1 = slim.conv2d(layer1, 64, [3, 3], stride=[2, 2], 
#                     normalizer_fn=slim.batch_norm,scope='conv_s_' + str(i))
                
#             top = slim.conv2d(layer1, num_classes, [3, 3], normalizer_fn=slim.batch_norm, 
#                 activation_fn=None, scope='conv_top')
#             top = slim.layers.flatten(top)
#             return {'Out' : top}
#         return io_fn( SISOTFM('ResNet', lambda : fn) )
#     return ResNet

