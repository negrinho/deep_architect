from deep_architect.contrib.misc.search_spaces.tensorflow.common import D, siso_tensorflow_module
import deep_architect.helpers.tensorflow as htf
import deep_architect.modules as mo

import tensorflow as tf

TFM = htf.TensorflowModule

def relu():
    def compile_fn(di, dh):
        def fn(di):
            return {'Out' : tf.nn.relu(di['In'])}
        return fn
    return siso_tensorflow_module('ReLU', compile_fn, {})

def pool_and_logits(num_classes):
    def compile_fn(di, dh):
        (_, height, width, _) = di['In'].get_shape().as_list()
        def fn(di):
            logits = tf.squeeze(tf.layers.conv2d(di['In'], num_classes, (height, width)))
            return {'Out': logits}
        return fn
    return TFM('Logits', {}, compile_fn, ['In'], ['Out']).get_io()

def global_pool_and_logits():
    def compile_fn(di, dh):
        (_, height, width, _) = di['In'].get_shape().as_list()
        def fn(di):
            logits = tf.layers.average_pooling2d(di['In'], (height, width), (1, 1))
            return {'Out': logits}
        return fn
    return TFM('Logits', {}, compile_fn, ['In'], ['Out']).get_io()


# TODO: perhaps add hyperparameters.
def batch_normalization():
    def compile_fn(di, dh):
        p_var = tf.placeholder(tf.bool)
        def fn(di):
            return {'Out' : tf.layers.batch_normalization(di['In'], training=p_var) }
        return fn, {p_var : 1}, {p_var : 0}
    return siso_tensorflow_module('BatchNormalization', compile_fn, {})

def avg_pool(h_kernel_size, h_stride):
    def compile_fn(di, dh):
        def fn(di):
            return {'Out' : tf.nn.avg_pool(di['In'],
                [1, dh['kernel_size'], dh['kernel_size'], 1], [1, dh['stride'], dh['stride'], 1], 'SAME')}
        return fn
    return siso_tensorflow_module('AvgPool', compile_fn, {
        'kernel_size' : h_kernel_size,
        'stride' : h_stride,
        })

# Add two inputs
def add():
    return htf.TensorflowModule('Add', {},
        lambda: lambda In0, In1: tf.add(In0, In1),
        ['In0', 'In1'], ['Out']).get_io()

def wrap_relu_batch_norm(io_pair, add_relu=True, add_bn=True):
    assert add_relu or add_bn
    elements = [True, add_relu, add_bn]
    module_fns = [
        lambda: io_pair,
        relu,
        batch_normalization]
    return mo.siso_sequential([module_fn() for i, module_fn in enumerate(module_fns) if elements[i]])

def wrap_batch_norm_relu(io_pair, add_relu=True, add_bn=True):
    assert add_relu or add_bn
    elements = [True, add_bn, add_relu]
    module_fns = [
        lambda: io_pair,
        batch_normalization,
        relu]
    return mo.siso_sequential([module_fn() for i, module_fn in enumerate(module_fns) if elements[i]])

