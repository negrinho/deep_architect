import tensorflow as tf 
from darch.contrib.search_spaces.tensorflow.common import D, siso_tfm
from darch.contrib.search_spaces.tensorflow.dnn import kaiming2015delving_initializer_conv, constant_initializer


def conv2d(h_num_filters, h_filter_width, h_stride, h_W_init_fn, h_b_init_fn):
    def cfn(di, dh , num_filters, filter_width, stride, W_init_fn, b_init_fn):
        (_, _, _, num_channels) = di['In'].get_shape().as_list()
        W = tf.Variable(W_init_fn([dh['filter_width'], dh['filter_width'], num_channels, dh['num_filters']]))
        b = tf.Variable(b_init_fn([dh['num_filters']]))
        def fn(di):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(di['In'], W, [1, dh['stride'], dh['stride'], 1], 'SAME'), b)}
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
    def cfn(di, dh):
        def fn(di):
            return {'Out' : tf.nn.max_pool(di['In'],
                [1, dh['kernel_size'], dh['kernel_size'], 1], [1, dh['stride'], dh['stride'], 1], 'SAME')}
        return fn
    return siso_tfm('MaxPool2D', cfn, {
        'kernel_size' : h_kernel_size, 'stride' : h_stride,})
