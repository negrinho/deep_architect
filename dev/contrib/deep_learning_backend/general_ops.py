from deep_architect.contrib.deep_learning_backend.backend import get_func


def dropout(h_keep_prob):
    fn = get_func('dropout')
    return fn(h_keep_prob=h_keep_prob)


def conv2d(h_num_filters, h_filter_width, h_stride, h_use_bias):
    fn = get_func('conv2d')
    return fn(h_num_filters=h_num_filters,
              h_filter_width=h_filter_width,
              h_stride=h_stride,
              h_use_bias=h_use_bias)


def max_pool2d(h_kernel_size, h_stride):
    fn = get_func('max_pool2d')
    return fn(h_kernel_size=h_kernel_size, h_stride=h_stride)


def avg_pool2d(h_kernel_size, h_stride):
    fn = get_func('avg_pool2d')
    return fn(h_kernel_size=h_kernel_size, h_stride=h_stride)


def batch_normalization():
    fn = get_func('batch_normalization')
    return fn()


def relu():
    fn = get_func('relu')
    return fn()


def global_pool2d():
    fn = get_func('global_pool2d')
    return fn()


def fc_layer(h_num_units):
    fn = get_func('fc_layer')
    return fn(h_num_units=h_num_units)
