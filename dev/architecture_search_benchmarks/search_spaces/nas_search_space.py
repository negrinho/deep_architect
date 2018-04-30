import darch.core as co
import darch.hyperparameters as hp
import darch.helpers.tensorflow as htf
import darch.modules as mo
import darch.contrib.useful.search_spaces.tensorflow.cnn2d as cnn2d
from darch.contrib.useful.search_spaces.tensorflow.common import D, siso_tfm
from common_ops import wrap_relu_batch_norm
import tensorflow as tf
import numpy as np
from collections import OrderedDict

TFM = htf.TFModule

# Take in array of boolean hyperparams, concatenate layers corresponding to true
# to form skip connections
def concatenate_skip_layers(h_connects):
    def cfn(di, dh):
        inputs = [di['In' + str(i)] for i in len(di) if dh['select_' + str(i)]]
        heights = [inp.get_shape().as_list()[1] for inp in inputs]
        widths = [inp.get_shape().as_list()[2] for inp in inputs]
        max_height = max(heights)
        max_width = max(widths)
        def fn(di):
            paddings = [tf.constant([
                [0,0], 
                [0, max_height - heights[i]],
                [0,  max_width - widths[i]], 
                [0,0]])]
            inputs = [tf.pad(inputs[i], paddings[i], 'CONSTANT')]
            return {'Out' : tf.concat(inputs, 3)}
        return fn
    return TFM('SkipConcat', 
               {'select_' + str(i) : h_connects[i] for i in range(len(h_connects))},
               cfn, ['In' + str(i) for i in len(h_connects)], ['Out']).get_io()

def conv2d_nas(h_num_filters, h_filter_height, h_filter_width, h_stride, h_use_bias):
    def cfn(di, dh):
        conv_op = tf.layers.Conv2D(dh['num_filters'], (dh['filter_height'], dh['width']),
            (dh['stride'],) * 2, use_bias=dh['use_bias'], padding='VALID')
        def fn(di):
            return {'Out' : conv_op(di['In'])}
        return fn
    return siso_tfm('Conv2DNAS', cfn, {
        'num_filters' : h_num_filters,
        'filter_height' : h_filter_height,
        'filter_width' : h_filter_width,
        'stride' : h_stride,
        'use_bias' : h_use_bias,
        })

def nas_repeat_fn(inputs, outputs):
    h_connects = [D([True, False]) for _ in outputs]
    skip_inputs, skip_outputs = concatenate_skip_layers(h_connects)
    for i, out_name in enumerate(outputs):
        outputs[out_name].connect(skip_inputs['In' + str(i)])
    conv_inputs, conv_outputs = conv2d_nas(D([24, 36, 48, 64]), 
                                           D([1, 3, 5, 7]), 
                                           D([1, 3, 5, 7]), 
                                           D([1, 2, 3]), 
                                           D([True]))
    wrap_inputs, wrap_outputs = wrap_relu_batch_norm((conv_inputs, conv_outputs))
    skip_outputs['Out'].connect(wrap_inputs['In'])
    outputs['Out' + str(len(outputs))] = wrap_outputs['Out']
    return inputs, outputs
    

def get_nas_search_space(num_classes):
    h_N = D(range(6, 21))
    mo.mimo_nested_repeat(mo.empty, nas_repeat_fn, h_N, ['In'], ['Out' + str(i) for ])