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

def siso_tfm(name, compile_fn, name_to_h={}, scope=None):
    return htf.TFModule(name, name_to_h, compile_fn, 
            ['In'], ['Out'], scope).get_io()

#trunc_normal_fn = lambda stddev: lambda shape: tf.truncated_normal(shape, stddev=stddev)
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

def relu():
    def cfn(di, dh):
        def fn(di):
            return {'Out' : tf.nn.relu(di['In'])}
        return fn
    return siso_tfm('ReLU', cfn)

def softmax():
    def cfn(di, dh):
        def fn(di):
            flat = tf.contrib.layers.flatten(di['In'])
            return {'Out': tf.nn.softmax(flat)}
        return fn
    return TFM('Softmax', {}, cfn, ['In'], ['Out']).get_io()

#def Affine(h_m, h_W_init_fn, h_b_init_fn):
#    def cfn(In, m, W_init_fn, b_init_fn):
#        shape = In.get_shape().as_list()
#        n = np.product(shape[1:])
#        W = tf.Variable( W_init_fn( [n, m] ) )
#        b = tf.Variable( b_init_fn( [m] ) )
#        def fn(In):
#            if len(shape) > 2:
#                In = tf.reshape(In, [-1, n])
#            Out = tf.add(tf.matmul(In, W), b)
#            # print In.get_shape().as_list()
#            return {'Out' : Out}
#        return fn
#    return siso_tfm('Affine', cfn, 
#        {'m' : h_m, 'W_init_fn' : h_W_init_fn, 'b_init_fn' : h_b_init_fn})

#def Dropout(h_keep_prob):
#    def cfn(keep_prob):
#        p = tf.placeholder(tf.float32)
#        def fn(In):
#            return {'Out' : tf.nn.dropout(In, p)} 
#        return fn, {p : keep_prob}, {p : 1.0} 
#    return siso_tfm('Dropout', cfn, {'keep_prob' : h_keep_prob})
    
# TODO: perhaps add hyperparameters.
def batch_normalization():
    def cfn(di, dh):
        p_var = tf.placeholder(tf.bool)
        def fn(di):
            return {'Out' : tf.layers.batch_normalization(di['In'], training=p_var) }
        return fn, {p_var : 1}, {p_var : 0}     
    return siso_tfm('BatchNormalization', cfn)

def Conv2D(h_num_filters, h_filter_size, h_stride, h_W_init_fn, h_b_init_fn):
    def cfn(di, dh):
        (_, height, width, channels) = di['In'].get_shape().as_list()
        W_init_fn = dh['W_init_fn']
        b_init_fn = dh['b_init_fn']

        W = tf.Variable( W_init_fn( [dh['filter_size'], dh['filter_size'], channels, dh['num_filters']] ) )
        b = tf.Variable( b_init_fn( [dh['num_filters']] ) )
        def fn(di):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(di['In'], W, [1, dh['stride'], dh['stride'], 1], 'SAME'), b)}
        return fn

    return siso_tfm('Conv2D', cfn, {
        'num_filters' : h_num_filters, 
        'filter_size' : h_filter_size, 
        'stride' : h_stride,
        'W_init_fn' : h_W_init_fn, 
        'b_init_fn' : h_b_init_fn,
        })

def max_pool(h_kernel_size, h_stride):
    def cfn(di, dh):
        def fn(di):
            return {'Out' : tf.nn.max_pool(di['In'], 
                [1, dh['kernel_size'], dh['kernel_size'], 1], [1, dh['stride'], dh['stride'], 1], 'SAME')}
        return fn
    return siso_tfm('MaxPool', cfn, {
        'kernel_size' : h_kernel_size, 
        'stride' : h_stride,
        })

def avg_pool(h_kernel_size, h_stride):
    def cfn(di, dh):
        def fn(di):
            return {'Out' : tf.nn.avg_pool(di['In'], 
                [1, dh['kernel_size'], dh['kernel_size'], 1], [1, dh['stride'], dh['stride'], 1], 'SAME')}
        return fn
    return siso_tfm('AvgPool', cfn, {
        'kernel_size' : h_kernel_size, 
        'stride' : h_stride,
        })

# Add two inputs
def add():
    return htf.TFModule('Add', {}, 
        lambda: lambda In0, In1: tf.add(In0, In1), 
        ['In0', 'In1'], ['Out']).get_io()

#def AffineSimplified(h_m):
#    def cfn(In, m):
#        shape = In.get_shape().as_list()
#        n = np.product(shape[1:])
#
#        def fn(In):
#            if len(shape) > 2:
#                In = tf.reshape(In, [-1, n])
#            return {'Out' : tf.layers.dense(In, m)}
#        return fn
#    return siso_tfm('AffineSimplified', cfn, {'m' : h_m})

#def Nonlinearity(h_or):
#    def cfn(idx):
#        def fn(In):
#            if idx == 0:
#                Out = tf.nn.relu(In)
#            elif idx == 1:
#                Out = tf.nn.relu6(In)
#            elif idx == 2:
#                Out = tf.nn.crelu(In)
#            elif idx == 3:
#                Out = tf.nn.elu(In)
#            elif idx == 4:
#                Out = tf.nn.softplus(In)
#            else:
#                raise ValueError
#            return {"Out" : Out}
#        return fn
#    return siso_tfm('Nonlinearity', cfn, {'idx' : h_or})

#def DNNCell(h_num_hidden, h_nonlin, h_swap, 
#        h_opt_drop, h_opt_bn, h_drop_keep_prob):
#    ms = [
#        AffineSimplified(h_num_hidden),
#        Nonlinearity(h_nonlin),
#        mo.SISOPermutation([
#            lambda: io_fn( 
#                mo.SISOOptional(
#                    lambda: io_fn( Dropout(h_drop_keep_prob) ), 
#                    h_opt_drop ) ),
#            lambda: io_fn( 
#                mo.SISOOptional(
#                    lambda: io_fn( BatchNormalization() ), 
#                    h_opt_bn ) ),
#        ], h_swap),
#        mo.Empty()
#    ]
#    ut.connect_sequentially(ms)
#    return io_lst_fn( ms )

#io_fn = lambda m: (m.inputs, m.outputs)
#io_lst_fn = lambda m_lst: (m_lst[0].inputs, m_lst[-1].outputs)
#
#def io_fn2(in0, in1, out):
#    return {'In0': in0, 'In1': in1}, {'Out': out}
#
#def ResidualSimplified(fn):
#    return mo.siso_residual(fn, lambda: io_fn( mo.Empty() ), lambda: add() )

# Basic convolutional module with preselected initialization function
def conv2D_simplified(h_num_filters, h_filter_size, h_stride):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        W_init_fn = kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        W = tf.Variable( W_init_fn( [dh['filter_size'], dh['filter_size'], channels, dh['num_filters']] ) )
        b = tf.Variable( b_init_fn( [dh['num_filters']] ) )
        def fn(di):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(di['In'], W, [1, dh['stride'], dh['stride'], 1], 'SAME'), b)}
        return fn

    return siso_tfm('Conv2DSimplified', cfn, {
        'num_filters' : h_num_filters,
        'stride' : h_stride,
        'filter_size' : h_filter_size})

# Basic convolutional module that selects number of filters based on number of
# input channels and the stride as in "Learning transferable architectures for scalable
# image recognition" (Zoph et al, 2017)
def conv2D(h_filter_size, h_stride):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        W_init_fn = kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        if(dh['stride'] == 2):
            num_filters = 2 * channels
        W = tf.Variable( W_init_fn( [dh['filter_size'], dh['filter_size'], channels, num_filters] ) )
        b = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(di):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(di['In'], W, [1, dh['stride'], dh['stride'], 1], 'SAME'), b)}
        return fn

    return siso_tfm('Conv2D', cfn, {
        'stride' : h_stride,
        'filter_size' : h_filter_size})

# Spatially separable convolutions, with output filter calculations as in Zoph17
def conv_spatial_separable(h_filter_size, h_stride):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        W_init_fn = kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        if(dh['stride'] == 2):
            num_filters = 2 * channels

        W1 = tf.Variable( W_init_fn( [1, dh['filter_size'], channels, num_filters] ) )
        b1 = tf.Variable( b_init_fn( [num_filters] ) )
        W2 = tf.Variable( W_init_fn( [dh['filter_size'], 1, num_filters, num_filters] ) )
        b2 = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(di):
            intermediate = tf.nn.bias_add(
                tf.nn.conv2d(di['In'], W1, [1, dh['stride'], dh['stride'], 1], 'SAME'), b1)
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(intermediate, W2, [1, 1, 1, 1], 'SAME'), b2)}
        return fn

    return siso_tfm('Conv2DSimplified', cfn, {
        'filter_size' : h_filter_size,
        'stride' : h_stride})
        

# Dilated convolutions, with output filter calculations as in Zoph17
def conv2D_dilated(h_filter_size, h_dilation, h_stride):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        W_init_fn = kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        if(dh['stride'] == 2):
            num_filters = 2 * channels

        W = tf.Variable( W_init_fn( [dh['filter_size'], dh['filter_size'], channels, num_filters] ) )
        b = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(di):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(di['In'], W, [1, dh['stride'], dh['stride'], 1], 'SAME', dilations=[1, dh['dilation'], dh['dilation'], 1]), b)}
        return fn

    return siso_tfm('Conv2DDilated', cfn, {
        'filter_size' : h_filter_size,
        'dilation' : h_dilation,
        'stride' : h_stride})


# Depth separable convolutions, with output filter calculations as in Zoph17
def conv2D_depth_separable(h_filter_size, h_channel_multiplier, h_stride):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        W_init_fn = kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        if(dh['stride'] == 2):
            num_filters = 2 * channels

        W_depth = tf.Variable( W_init_fn( [dh['filter_size'], dh['filter_size'], channels, dh['channel_multiplier']] ) )
        W_point = tf.Variable( W_init_fn( [1, 1, channels * dh['channel_multiplier'], num_filters] ) )
        b = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(di):
            return {'Out' : tf.nn.bias_add(
                tf.nn.separable_conv2d(di['In'], W_depth, W_point, [1, dh['stride'], dh['stride'], 1], 'SAME', [1, 1]), b)}
        return fn

    return siso_tfm('Conv2DSeparable', cfn, {
        'filter_size' : h_filter_size,
        'channel_multiplier' : h_channel_multiplier,
        'stride' : h_stride})

def conv1x1(h_num_filters):
    return conv2D_simplified(h_num_filters, D([1]), D([1]))

# A module that wraps an io pair with relu at the before and batch norm after
def wrap_relu_batch_norm(io_pair):
    r_inputs, r_outputs = relu()
    b_inputs, b_outputs = batch_normalization()
    r_outputs['Out'].connect(io_pair[0]['In'])
    io_pair[1]['Out'].connect(b_inputs['In'])
    return r_inputs, b_outputs

# The operations used in search space 1 and search space 3 in Regularized Evolution for
# Image Classifier Architecture Search (Real et al, 2018)
def sp1_operation(h_op_name, h_stride):
    return mo.siso_or({
            'identity': lambda: mo.empty(),
            'd_sep3': lambda: wrap_relu_batch_norm(conv2D_depth_separable(D([3]), D([1]), h_stride)),
            'd_sep5': lambda: wrap_relu_batch_norm(conv2D_depth_separable(D([5]), D([1]), h_stride)),
            'd_sep7': lambda: wrap_relu_batch_norm(conv2D_depth_separable(D([7]), D([1]), h_stride)),
            'avg3': lambda: avg_pool(D([3]), h_stride),
            'max3': lambda: max_pool(D([3]), h_stride),
            'dil3': lambda: wrap_relu_batch_norm(conv2D_dilated(D([3]), D([2]), h_stride)),
            's_sep7': lambda: wrap_relu_batch_norm(conv_spatial_separable(D([7]), h_stride))
            }, h_op_name)

# The operations used in search space 2 in Regularized Evolution for
# Image Classifier Architecture Search (Real et al, 2018)
def sp2_operation(h_op_name, h_stride):
    return mo.siso_or({
            'identity': lambda: mo.empty(),
            'conv1': lambda: wrap_relu_batch_norm(conv2D(D([1]), h_stride)),
            'conv3': lambda: wrap_relu_batch_norm(conv2D(D([3]), h_stride)),
            'd_sep3': lambda: wrap_relu_batch_norm(conv2D_depth_separable(D([3]), D([1]), h_stride)),
            'd_sep5': lambda: wrap_relu_batch_norm(conv2D_depth_separable(D([5]), D([1]), h_stride)),
            'd_sep7': lambda: wrap_relu_batch_norm(conv2D_depth_separable(D([7]), D([1]), h_stride)),
            'avg2': lambda: avg_pool(D([2]), h_stride),
            'avg3': lambda: avg_pool(D([3]), h_stride),
            'min2': lambda: mo.empty(),
            'max2': lambda: max_pool(D([2]), h_stride),
            'max3': lambda: max_pool(D([3]), h_stride),
            'dil3': lambda: wrap_relu_batch_norm(conv2D_dilated(D([3]), D([2]), h_stride)),
            'dil5': lambda: wrap_relu_batch_norm(conv2D_dilated(D([5]), D([2]), h_stride)),
            'dil7': lambda: wrap_relu_batch_norm(conv2D_dilated(D([7]), D([2]), h_stride)),
            's_sep3': lambda: wrap_relu_batch_norm(conv_spatial_separable(D([3]), h_stride)),
            's_sep7': lambda: wrap_relu_batch_norm(conv_spatial_separable(D([7]), h_stride)),
            'dil3_2': lambda: wrap_relu_batch_norm(conv2D_dilated(D([3]), D([2]), h_stride)),
            'dil3_4': lambda: wrap_relu_batch_norm(conv2D_dilated(D([3]), D([4]), h_stride)),
            'dil3_6': lambda: wrap_relu_batch_norm(conv2D_dilated(D([3]), D([6]), h_stride))
            }, h_op_name)

# A module that takes in a specifiable number of inputs, uses 1x1 convolutions to make the number
# filters match, and then adds the inputs together
def mi_add(num_terms):
    def cfn(di, dh):
        (_, _, _, min_channels) = di['In0'].get_shape().as_list()
        for i in range(num_terms):
            (_, _, _, channels) = di['In' + str(i)].get_shape().as_list()
            min_channels = channels if channels < min_channels else min_channels

        W_init_fn = kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)
        w_dict = {}
        b_dict = {}
        for i in range(num_terms):
            (_, _, _, channels) = di['In' + str(i)].get_shape().as_list()
            if channels != min_channels:
                w_dict[i] = tf.Variable( W_init_fn( [1, 1, channels, min_channels]))
                b_dict[i] = tf.Variable( b_init_fn( [min_channels]))

        def fn(In):
            trans = []
            for i in range(num_terms):
                if i in w_dict:
                    trans.append(tf.nn.bias_add(tf.nn.conv2d(di['In' + str(i)], w_dict[i], [1, 1, 1, 1], 'SAME'), b_dict[i]))
                else:
                    trans.append(di['In' + str(i)])
            total_sum = trans[0]
            for i in range(1, num_terms):
                total_sum = tf.add(total_sum, trans[i])
            return {'Out' : total_sum}
        return fn
    return TFM('MultiInputAdd', {}, cfn, ['In' + str(i) for i in xrange(num_terms)], ['Out']).get_io()

# A module that applies a sp1_operation to two inputs, and adds them together
def sp1_combine(h_op1_name, h_op2_name, h_stride):
    in1 = lambda: sp1_operation(h_op1_name, h_stride)
    in2 = lambda: sp1_operation(h_op2_name, h_stride)
    return mo.mimo_combine([in1, in2], mi_add)

# A module that selects an input from a list of inputs
def selector(h_selection, sel_fn, num_selections):
    def cfn(di, dh):
        selection = dh['selection']
        sel_fn(selection)
        def fn(di):
            return {'Out': di['In' + str(selection)]}
        return fn
    return TFM('Selector', {'selection': h_selection}, cfn, ['In' + str(i) for i in xrange(num_selections)], ['Out']).get_io()

# A module that outlines the basic cell structure in Learning Transferable 
# Architectures for Scalable Image Recognition (Zoph et al, 2017)
def basic_cell(h_sharer, C=5, normal=True):
    i_inputs, i_outputs = mo.empty(num_connections=2)
    available = [i_outputs['Out' + str(i)] for i in xrange(len(i_outputs))]
    unused = [False] * (C + 2)
    for i in xrange(C):
        h_in0_pos = h_sharer.get('h_in0_pos_' + str(i))
        h_in1_pos = h_sharer.get('h_in1_pos_' + str(i))

        def select(selection):
            unused[selection] = True

        sel0_inputs, sel0_outputs = selector(h_in0_pos, select, len(available))
        sel1_inputs, sel1_outputs = selector(h_in1_pos, select, len(available))

        h_op1 = h_sharer.get('h_op1_' + str(i))
        h_op2 = h_sharer.get('h_op2_' + str(i))
        
        for o_idx in xrange(len(available)):
            available[o_idx].connect(sel0_inputs['In' + str(o_idx)])
            available[o_idx].connect(sel1_inputs['In' + str(o_idx)])
        
        comb_inputs, comb_outputs = sp1_combine(h_op1, h_op2, D([1 if normal or i > 0 else 2]))
        sel0_outputs['Out'].connect(comb_inputs['In0'])
        sel1_outputs['Out'].connect(comb_inputs['In1'])
        available.append(comb_outputs['Out'])
    
    unused_outs = [available[i] for i in xrange(len(unused)) if not unused[i]]
    s_inputs, s_outputs = mi_add(len(unused_outs))
    for i in range(len(unused_outs)):
        unused_outs[i].connect(s_inputs['In' + str(i)])
    return i_inputs, s_outputs

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

# A module that creates a number of repeated cells (both reduction and normal)
# as in Learning Transferable Architectures for Scalable Image Recognition (Zoph et al, 2017)
def ss_repeat(h_N, h_sharer, C, num_ov_repeat, scope=None):
    def sub_fn(N):
        assert N > 0
        i_inputs, i_outputs = mo.simo_split(2)
        prev_2 = [i_outputs['Out0'], i_outputs['Out1']]

        for i in range(num_ov_repeat):
            for j in range(N):
                print '(%d, %d)' %(i,  j)
                norm_inputs, norm_outputs = basic_cell(h_sharer, C=C, normal=True)
                prev_2[0].connect(norm_inputs['In0'])
                prev_2[1].connect(norm_inputs['In1'])
                prev_2[0] = prev_2[1]
                prev_2[1] = norm_outputs['Out']
            if j < num_ov_repeat - 1:
                red_inputs, red_outputs = basic_cell(h_sharer, C=C, normal=False)
                prev_2[0].connect(red_inputs['In0'])
                prev_2[1].connect(red_inputs['In1'])
                prev_2[0] = prev_2[1]
                prev_2[1] = red_outputs['Out']
        
        soft_inputs, soft_outputs = softmax()
        prev_2[1].connect(soft_inputs['In'])
        return i_inputs, soft_outputs
    return mo.substitution_module('SS_repeat', {'N': h_N}, sub_fn, ['In'], ['Out'], scope)

# Search space 1 from Regularized Evolution for Image Classifier Architecture Search (Real et al, 2018)
def get_search_space_1(num_classes):
    co.Scope.reset_default_scope()
    C = 5
    h_N = D([4, 6])
    h_F = D([32, 64, 128])
    h_sharer = hp.HyperparameterSharer()
    for i in xrange(C):
        h_sharer.register('h_op1_' + str(i), lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 'dil3', 's_sep7']))
        h_sharer.register('h_op2_' + str(i), lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 'dil3', 's_sep7']))
        h_sharer.register('h_in0_pos_' + str(i), lambda: D([range(2 + i)]))
        h_sharer.register('h_in1_pos_' + str(i), lambda: D([range(2 + i)]))
    i_inputs, i_outputs = conv1x1(h_F)
    o_inputs, o_outputs = ss_repeat(h_N, h_sharer, C, 3)
    i_outputs['Out'].connect(o_inputs['In'])
    r_inputs, r_outputs = mo.siso_sequential([mo.empty(), (i_inputs, o_outputs), mo.empty()])
    return r_inputs, r_outputs, hyperparameters_fn()


# Search space 3 from Regularized Evolution for Image Classifier Architecture Search (Real et al, 2018)
def get_search_space_3(num_classes):
    co.Scope.reset_default_scope()
    C = 15
    h_N = D([4, 6])
    h_F = D([32, 64, 128])
    h_sharer = hp.HyperparameterSharer()
    for i in xrange(C):
        h_sharer.register('h_op1_' + str(i), lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 'dil3', 's_sep7']))
        h_sharer.register('h_op2_' + str(i), lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 'dil3', 's_sep7']))
        h_sharer.register('h_in0_pos_' + str(i), lambda: D([range(2 + i)]))
        h_sharer.register('h_in1_pos_' + str(i), lambda: D([range(2 + i)]))
    i_inputs, i_outputs = conv1x1(h_F)
    o_inputs, o_outputs = ss_repeat(h_N, h_sharer, C, 3)
    i_outputs['Out'].connect(o_inputs['In'])
    r_inputs, r_outputs = mo.siso_sequential([mo.empty(), (i_inputs, o_outputs), mo.empty()])
    return r_inputs, r_outputs, hyperparameters_fn()

# Search space 2 from Regularized Evolution for Image Classifier Architecture Search (Real et al, 2018)
def get_search_space_2(num_classes):
    co.Scope.reset_default_scope()
    C = 5
    h_N = D([4, 6])
    h_F = D([32, 64, 128])
    h_sharer = hp.HyperparameterSharer()
    for i in xrange(C):
        h_sharer.register('h_op1_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
          'min2', 'max2', 'max3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6']))
        h_sharer.register('h_op2_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
          'min2', 'max2', 'max3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6']))
        h_sharer.register('h_in0_pos_' + str(i), lambda: D([range(2 + i)]))
        h_sharer.register('h_in1_pos_' + str(i), lambda: D([range(2 + i)]))
    i_inputs, i_outputs = conv1x1(h_F)
    o_inputs, o_outputs = ss_repeat(h_N, h_sharer, C, 3)
    i_outputs['Out'].connect(o_inputs['In'])
    return i_inputs, o_outputs, hyperparameters_fn()

#def get_ss0_fn(num_classes):
#    def fn():
#        def cell_fn():
#            h_num_hidden = D([ 64, 128, 256, 512, 1024])
#            h_nonlin = D([ 0, 1, 2, 3, 4 ])
#            h_swap = D([ 0, 1 ])
#            h_opt_drop = D([ 0, 1 ])
#            h_opt_bn = D([ 0, 1 ])
#            h_drop_keep_prob = D([ 0.3, 0.5, 0.7 ])
#
#            return DNNCell(h_num_hidden, h_nonlin, h_swap, 
#                h_opt_drop, h_opt_bn, h_drop_keep_prob)
#
#        co.Scope.reset_default_scope()
#        ms = [
#            mo.Empty(),
#            mo.SISORepeat( lambda: io_fn( 
#                mo.SISORepeat(cell_fn, 
#                D([1, 2, 4]) ) ), 
#            D([1, 2, 4]) ),
#            AffineSimplified( D([ num_classes ]) )
#        ]
#        ut.connect_sequentially(ms)
#
#        return ms[0].inputs, ms[-1].outputs, hyperparameters_fn()
#    return fn
#
#def get_ss1_fn(num_classes):
#    def fn():
#        co.Scope.reset_default_scope()
#
#        h_m = D([ num_classes ])
#        h_W_init = D([ xavier_initializer_affine() ])
#        h_b_init = D([ const_fn(0.0) ])
#
#        ms = [
#            Affine(h_m, h_W_init, h_b_init)
#        ]
#        ut.connect_sequentially(ms)
#        
#        return ms[0].inputs, ms[-1].outputs, hyperparameters_fn()
#    return fn
#

#def get_ss2_fn(num_classes):
#    def fn():
#        co.Scope.reset_default_scope()
#
#        def fn1(h_num_filters, h_filter_size, h_perm, h_opt):
#            ms = [
#                Conv2DSimplified(h_num_filters, h_filter_size, 1),
#                mo.SISOPermutation([
#                    lambda: io_fn( ReLU() ), 
#                    lambda: io_fn( mo.SISOOptional(
#                        lambda: io_fn( BatchNormalization() ), h_opt ) )
#                ], h_perm),
#            ]
#            ut.connect_sequentially(ms)
#            return ms[0].inputs, ms[-1].outputs
#
#        def fn2(h_num_filters, h_filter_size, h_perm, h_opt, h_or):
#            f = lambda: fn1(h_num_filters, h_filter_size, h_perm, h_opt)
#            return io_fn( 
#                mo.SISOOr([
#                    f, f
#                    # lambda: ResidualSimplified(f)  
#                ], h_or) )
#
#        def fn3(hr_num_filters, hr_filter_size, ho_num_filters, ho_filter_size, h_opt, h_or):
#            ms = [
#                mo.SISOOr([
#                    lambda: io_fn( Conv2DSimplified(hr_num_filters, hr_filter_size, 2) ),
#                    lambda: io_fn( MaxPool(hr_filter_size, D([ 2 ])) ) ], h_or),
#                mo.SISOOptional( 
#                    lambda: io_fn( Conv2DSimplified(ho_num_filters, ho_filter_size, 1) ), h_opt ),
#            ]
#            ut.connect_sequentially(ms)
#
#            return io_lst_fn( ms )        
#
#        h_f = D([ 32, 64, 128 ])
#        h_rep = D([ 1, 2, 4, 8 ])
#        h_perm = hp.Bool()
#        h_or = hp.Bool()
#        h_opt = hp.Bool()
#        fs = D([ 3 ])
#
#        f1 = lambda: fn2(h_f, fs, h_perm, h_opt, h_or)
#        f2 = lambda: fn2(h_f, fs, h_perm, h_opt, h_or)
#        f3 = lambda: fn2(h_f, fs, h_perm, h_opt, h_or)
#
#        h3_or = hp.Bool()
#        h3_opt = hp.Bool()
#
#        lst = [
#            io_fn( mo.Empty() ),
#            fn3(h_f, fs, h_f, fs, h3_opt, h3_or),
#            io_fn( mo.SISORepeat(f1, h_rep) ),
#            fn3(h_f, fs, h_f, fs, h3_opt, h3_or),
#            io_fn( mo.SISORepeat(f2, h_rep) ),
#            fn3(h_f, fs, h_f, fs, h3_opt, h3_or),
#            io_fn( mo.SISORepeat(f3, h_rep) ),
#            fn3(h_f, fs, h_f, fs, h3_opt, h3_or),
#            io_fn( AffineSimplified(D([ 2 ])) )
#        ]
#
#        for (ins_prev, outs_prev), (ins_next, outs_next) in zip(lst[:-1], lst[1:]):
#            outs_prev['Out'].connect( ins_next['In'] )
#
#        return lst[0][0], lst[-1][1], hyperparameters_fn()
#    return fn

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

