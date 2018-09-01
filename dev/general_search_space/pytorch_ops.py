from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from darch.contrib.useful.search_spaces.pytorch.common import siso_pytorch_module

def calculate_same_padding(h_in, w_in, stride, filter_size):
    h_out = ceil(float(h_in) / float(stride))
    w_out = ceil(float(w_in) / float(stride))
    if (h_in % stride == 0):
        pad_along_height = max(filter_size - stride, 0)
    else:
        pad_along_height = max(filter_size - (h_in % stride), 0)
    if (h_in % stride == 0):
        pad_along_width = max(filter_size - stride, 0)
    else:
        pad_along_width = max(filter_size - (h_in % stride), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)

def conv2d(h_num_filters, h_filter_size, h_stride, h_use_bias):
    def cfn(di, dh):
        (_, channels, height, width) = di['In'].size()
        padding = nn.ZeroPad2d(
            calculate_same_padding(height, width, dh['stride'], dh['filter_size']))
        conv = nn.Conv2d(
            channels, dh['num_filters'], dh['filter_size'],
            stride=dh['stride'], bias=dh['use_bias'])
        def fn(di):
            x = padding(di['In'])
            return {'Out' : conv(x)}
        return fn, [conv, padding]
    return siso_pytorch_module('Conv2D', cfn, {
        'num_filters' : h_num_filters,
        'filter_size' : h_filter_size,
        'stride' : h_stride,
        'use_bias' : h_use_bias,
        })

def max_pool2d(h_kernel_size, h_stride):
    def cfn(di, dh):
        (_, _, height, width) = di['In'].size()
        padding = nn.ZeroPad2d(
            calculate_same_padding(height, width, dh['stride'], dh['kernel_size']))
        max_pool = nn.MaxPool2d(dh['kernel_size'], stride=dh['stride'])
        def fn(di):
            x = padding(di['In'])
            return {'Out' : max_pool(x)}
        return fn, [padding, max_pool]
    return siso_pytorch_module('MaxPool2D', cfn, {
        'kernel_size' : h_kernel_size, 'stride' : h_stride})

def dropout(h_keep_prob):
    def cfn(di, dh):
        dropout_layer = nn.Dropout(p=dh['keep_prob'])
        def fn(di):
            return {'Out' : dropout_layer(di['In'])}
        return fn, [dropout_layer]
    return siso_pytorch_module('Dropout', cfn, {'keep_prob' : h_keep_prob})

def batch_normalization():
    def cfn(di, dh):
        (_, channels, _, _) = di['In'].size()
        batch_norm = nn.BatchNorm2d(channels)
        def fn(di):
            return {'Out' : batch_norm(di['In'])}
        return fn, [batch_norm]
    return siso_pytorch_module('BatchNormalization', cfn, {})

def relu():
    def cfn(di, dh):
        def fn(di):
            return {'Out' : F.relu(di['In'])}
        return fn, []
    return siso_pytorch_module('ReLU', cfn, {})

def global_pool2d():
    def cfn(di, dh):
        (_, _, height, width) = di['In'].size()
        def fn(di):
            x = F.avg_pool2d(di['In'], (height, width))
            x = torch.squeeze(x, 2)
            return {'Out' : torch.squeeze(x, 2)}
        return fn, []
    return siso_pytorch_module('GlobalAveragePool', cfn, {})

def fc_layer(h_num_units):
    def cfn(di, dh):
        (_, channels) = di['In'].size()
        fc = nn.Linear(channels, dh['num_units'])
        def fn(di):
            return {'Out' : fc(di['In'])}
        return fn, [fc]
    return siso_pytorch_module('FCLayer', cfn, {'num_units' : h_num_units})

func_dict = {
    'dropout': dropout,
    'conv2d': conv2d,
    'max_pool2d': max_pool2d,
    'batch_normalization': batch_normalization,
    'relu': relu,
    'global_pool2d': global_pool2d,
    'fc_layer': fc_layer
}
