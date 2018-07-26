from darch.contrib.useful.search_spaces.tensorflow.common import siso_tfm 
import tensorflow as tf

def conv2d(h_num_filters, h_filter_width, h_stride, h_use_bias):
    def cfn(di, dh):
        conv_op = tf.layers.Conv2D(dh['num_filters'], (dh['filter_width'],) * 2,
            (dh['stride'],) * 2, use_bias=dh['use_bias'], padding='SAME')
        def fn(di):
            return {'Out' : conv_op(di['In'])}
        return fn
    return siso_tfm('Conv2D', cfn, {
        'num_filters' : h_num_filters,
        'filter_width' : h_filter_width,
        'stride' : h_stride,
        'use_bias' : h_use_bias,
        })

def max_pool2d(h_kernel_size, h_stride):
    def cfn(di, dh):
        def fn(di):
            return {'Out' : tf.nn.max_pool(di['In'],
                [1, dh['kernel_size'], dh['kernel_size'], 1], [1, dh['stride'], dh['stride'], 1], 'SAME')}
        return fn
    return siso_tfm('MaxPool2D', cfn, {
        'kernel_size' : h_kernel_size, 'stride' : h_stride,})

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

def relu():
    return siso_tfm('ReLU', lambda di, dh: lambda di: {'Out' : tf.nn.relu(di['In'])}, {})

def global_pool2d():
    def cfn(di, dh):
        def fn(di):
            return {'Out' : tf.reduce_mean(di['In'], [1,2])}
        return fn
    return siso_tfm('GlobalAveragePool', cfn, {})

def fc_layer(h_num_units):
    def cfn(di, dh):
        fc = tf.layers.Dense(dh['num_units'])
        def fn(di):
            return {'Out' : fc(di['In'])}
        return fn
    return siso_tfm('FCLayer', cfn, {'num_units' : h_num_units})

func_dict = {
    'dropout': dropout,
    'conv2d': conv2d,
    'max_pool2d': max_pool2d,
    'batch_normalization': batch_normalization,
    'relu': relu,
    'global_pool2d': global_pool2d,
    'fc_layer': fc_layer
}