import darch.core as co
import darch.hyperparameters as hps
import darch.utils as ut
import darch.tensorflow_helpers as tf_helpers
import darch.searchers as se
import darch.modules as mo
import darch.helpers.dynet as dyh

import tensorflow as tf
import numpy as np

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


import darch.core as co
import darch.hyperparameters as hps
import darch.utils as ut
import darch.tensorflow_helpers as tf_helpers
import darch.searchers as se
import tensorflow as tf
import numpy as np

TFM = tf_helpers.TFModule
D = hps.Discrete

def get_search_space_fn(num_labels):
    def search_space_fn():
        xs = [
            Conv2D(
                D([ 32 ]),
                D([ 5 ]),
                D([ 1 ]),
                D([ trunc_normal_fn(0.1) ]),
                D([ const_fn(0.0) ])),
            ReLU(),
            MaxPool(
                D([ 2 ]), 
                D([ 2 ]) ),
            Conv2D(
                D([ 64 ]),
                D([ 5 ]),
                D([ 1 ]),
                D([ trunc_normal_fn(0.1) ]),
                D([ const_fn(0.1) ])),
            ReLU(),
            MaxPool(
                D([ 2 ]), 
                D([ 2 ]) ),
            Affine(
                D([ 512 ]),
                D([ trunc_normal_fn(0.1) ]),
                D([ const_fn(0.1) ])),
            ReLU(),
            Dropout(
                D([ 0.5 ])),
            Affine(
                D([ 10 ]),
                D([ trunc_normal_fn(0.1) ]),
                D([ const_fn(0.1) ]))
        ]

        ut.connect_sequentially(xs)
        inputs, outputs = xs[0].inputs, xs[-1].outputs
        return inputs, outputs, {}
    return search_space_fn

def get_ref_model(data_node):
    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.global_variables_initializer().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED, dtype=data_type()))
    conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
    conv2_weights = tf.Variable(tf.truncated_normal(
        [5, 5, 32, 64], stddev=0.1,
        seed=SEED, dtype=data_type()))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                            stddev=0.1,
                            seed=SEED,
                            dtype=data_type()))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                    stddev=0.1,
                                                    seed=SEED,
                                                    dtype=data_type()))
    fc2_biases = tf.Variable(tf.constant(
        0.1, shape=[NUM_LABELS], dtype=data_type()))
    aff_params.extend([fc1_weights, fc1_biases, fc2_weights, fc2_biases])

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data_node,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.

    p = tf.placeholder(tf.float32)
    hidden = tf.nn.dropout(hidden, p, seed=SEED)    
    logits = tf.matmul(hidden, fc2_weights) + fc2_biases
    return logits, {p : 0.5}, {p : 1.0}, aff_params
