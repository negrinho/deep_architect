from darch.contrib.useful.search_spaces.tensorflow.common import D, siso_tfm
import darch.helpers.tensorflow as htf

import tensorflow as tf

TFM = htf.TFModule

def relu():
    def cfn(di, dh):
        def fn(di):
            return {'Out' : tf.nn.relu(di['In'])}
        return fn
    return siso_tfm('ReLU', cfn, {})

def pool_and_logits(num_classes):
    def cfn(di, dh):
        (_, height, width, _) = di['In'].get_shape().as_list()
        def fn(di):
            logits = tf.squeeze(tf.layers.conv2d(di['In'], num_classes, (height, width)))
            return {'Out': logits}
        return fn
    return TFM('Logits', {}, cfn, ['In'], ['Out']).get_io()

# TODO: perhaps add hyperparameters.
def batch_normalization():
    def cfn(di, dh):
        p_var = tf.placeholder(tf.bool)
        def fn(di):
            return {'Out' : tf.layers.batch_normalization(di['In'], training=p_var) }
        return fn, {p_var : 1}, {p_var : 0}
    return siso_tfm('BatchNormalization', cfn, {})

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

def wrap_relu_batch_norm(io_pair):
    r_inputs, r_outputs = relu()
    b_inputs, b_outputs = batch_normalization()
    r_outputs['Out'].connect(io_pair[0]['In'])
    io_pair[1]['Out'].connect(b_inputs['In'])
    return r_inputs, b_outputs

