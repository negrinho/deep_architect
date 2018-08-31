from dev.general_search_space.backend import get_func

def dropout(h_keep_prob):
    fn = get_func('dropout')
    return fn(h_keep_prob)

def conv2d(h_num_filters, h_filter_size, h_stride, h_use_bias):
    fn = get_func('conv2d')
    return fn(h_num_filters, h_filter_size, h_stride, h_use_bias)

def batch_normalization():
    fn = get_func('batch_normalization')
    return fn()

def relu():
    fn = get_func('relu')
    return fn()

def max_pool2d(h_kernel_size, h_stride):
    fn = get_func('max_pool2d')
    return fn(h_kernel_size, h_stride)

def global_pool2d():
    fn = get_func('global_pool2d')
    return fn()

def fc_layer(h_num_units):
    fn = get_func('fc_layer')
    return fn(h_num_units)
