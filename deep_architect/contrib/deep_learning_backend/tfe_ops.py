from deep_architect.helpers.tfeager import siso_tfeager_module
import tensorflow as tf

def max_pool2d(h_kernel_size, h_stride):
    def cfn(di, dh):
        def fn(di, isTraining=True):
            return {'Out' : tf.nn.max_pool(di['In'],
                [1, dh['kernel_size'], dh['kernel_size'], 1], [1, dh['stride'], dh['stride'], 1], 'SAME')}
        return fn
    return siso_tfeager_module('MaxPool2D', cfn, {
        'kernel_size' : h_kernel_size,
        'stride' : h_stride,})

def batch_normalization():
    def cfn(di, dh):
        def fn(di, isTraining=True):
            return {'Out' : tf.layers.batch_normalization(di['In'], training=isTraining)}
        return fn
    return siso_tfeager_module('BatchNormalization', cfn, {})

def relu():
    def cfn(di, dh):
        def fn(di, isTraining=True):
            return {'Out' : tf.nn.relu(di['In'])}
        return fn
    return siso_tfeager_module('ReLU', cfn, {})

def conv2d(h_num_filters, h_filter_width, h_stride, h_dilation_rate, h_use_bias):
    def cfn(di, dh):
        conv = tf.layers.Conv2D(dh['num_filters'], dh['filter_width'],
            dh['stride'], dilation_rate=dh['dilation_rate'], use_bias=dh['use_bias'], padding='SAME')
        def fn(di, isTraining=True):
            return {'Out' : conv(di['In'])}
        return fn
    return siso_tfeager_module('Conv2D', cfn, {
        'num_filters' : h_num_filters,
        'filter_width' : h_filter_width,
        'stride' : h_stride,
        'use_bias' : h_use_bias,
        'dilation_rate' : h_dilation_rate
        })

def separable_conv2d(h_num_filters, h_filter_width, h_stride, h_dilation_rate,
                     h_depth_multiplier, h_use_bias):
    def cfn(di, dh):
        conv_op = tf.layers.SeparableConv2D(dh['num_filters'], dh['filter_width'],
            strides=dh['stride'], dilation_rate=dh['dilation_rate'],
            depth_multiplier=dh['depth_multiplier'], use_bias=dh['use_bias'],
            padding='SAME')
        def fn(di, isTraining=True):
            return {'Out' : conv_op(di['In'])}
        return fn
    return siso_tfeager_module('SeparableConv2D', cfn, {
        'num_filters' : h_num_filters,
        'filter_width' : h_filter_width,
        'stride' : h_stride,
        'use_bias' : h_use_bias,
        'dilation_rate' : h_dilation_rate,
        'depth_multiplier' : h_depth_multiplier,
        })

def avg_pool2d(h_kernel_size, h_stride):
    def cfn(di, dh):
        def fn(di, isTraining=True):
            return {'Out' : tf.nn.avg_pool(di['In'],
                [1, dh['kernel_size'], dh['kernel_size'], 1], [1, dh['stride'], dh['stride'], 1], 'SAME')}
        return fn
    return siso_tfeager_module('MaxPool2D', cfn, {
        'kernel_size' : h_kernel_size, 'stride' : h_stride,})

def dropout(h_keep_prob):
    def cfn(di, dh):
        def fn(di, isTraining=True):
            out = tf.nn.dropout(di['In'], dh['keep_prob'] if isTraining else 1.0)
            return {'Out': out}
            # return {'Out' : di['In']}
        return fn
    return siso_tfeager_module('Dropout', cfn, {'keep_prob' : h_keep_prob})

def global_pool2d():
    def cfn(di, dh):
        def fn(di, isTraining=True):
            return {'Out' : tf.reduce_mean(di['In'], [1,2])}
        return fn
    return siso_tfeager_module('GlobalAveragePool', cfn, {})

def fc_layer(h_num_units):
    def cfn(di, dh):
        fc = tf.layers.Dense(dh['num_units'])
        def fn(di, isTraining=True):
            return {'Out' : fc(di['In'])}
        return fn
    return siso_tfeager_module('FCLayer', cfn, {'num_units' : h_num_units})

func_dict = {
    'dropout': dropout,
    'conv2d': conv2d,
    'max_pool2d': max_pool2d,
    'batch_normalization': batch_normalization,
    'relu': relu,
    'global_pool2d': global_pool2d,
    'fc_layer': fc_layer
}