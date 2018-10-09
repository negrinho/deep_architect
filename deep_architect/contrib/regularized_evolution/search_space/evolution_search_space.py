###${MARKDOWN}
# This search space is the one used in
# [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) and
# [Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/abs/1802.01548),
# for the purposes of reproducing the
# [Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/abs/1802.01548)
# paper. The search space was created using descriptions in the paper and examining
# the code from [here](https://github.com/tensorflow/tpu/tree/master/models/official/amoeba_net).
# While the search space does not match the one used in the original work
# exactly, it contains most of the major ideas used. We have annotated this file
# to show how to implement even fairly complex search spaces in our framework.
#
# To give a high level summary of the search space, it essentially specifies two
# types of cells--Normal cells and Reduction cells--which are then stacked to
# create the overall architecture. Each cell takes in as inputs the outputs of
# the previous two layers. Normal cells maintain the number of filters and the
# kernel size, which Reduction cells double the number of filters while halving
# the kernel size in both dimensions. N normal cells are followed by 1 reduction
# cell and this pattern continues (except for the very end, where the reduction
# cell is omitted), a certain number of times that is constant for the whole
# search. The initial number of filters used is parameterized as F. Each
# individual cell is constructed as follows: two of the previous inputs available
# to the cell are independently chosen, after which each is transformed using
# one of a certain set of prespecified operations (conv, maxpool...). The two
# transformed inputs are added together and this new value is added to the list
# of inputs. This is done C times, where C is another parameter, and then the
# unused inputs in the list are added up together to form the output of the cell.
# For reduction cells, any time one of the original two inputs is used, a stride
# of two is applied.
#
# This tutorial assumes familiarity with the rest of the DeepArchitect framework
# including substitution modules and dependent hyperparameters. It is meant to
# showcase how these components can be used to create a complex search space. We
# will mostly be using modules adhering to framework agnostic signatures, but
# some specific modules will be used that are Tensorflow specific.
# (Note, in this tutorial, we will refer to modules using framework agnostic
# signatures as framework agnostic modules, even though many are Tensorflow
# specific modules specified in tf_ops)

from __future__ import absolute_import
from builtins import str
from builtins import range

import tensorflow as tf
from collections import OrderedDict

import deep_architect.core as co
import deep_architect.hyperparameters as hp
import deep_architect.helpers.tensorflow as htf
import deep_architect.modules as mo
import deep_architect.contrib.misc.search_spaces.tensorflow.cnn2d as cnn2d
from deep_architect.contrib.deep_learning_backend.tf_ops import (
    relu, batch_normalization, conv2d, separable_conv2d, avg_pool2d, max_pool2d,
    fc_layer, global_pool2d)

from deep_architect.hyperparameters import Discrete as D

TFM = htf.TensorflowModule
const_fn = lambda c: lambda shape: tf.constant(c, shape=shape)


# This is a module created using framework agnostic modules to perform spatially
# separable convolutions (eg a 1x3 conv followed by a 3x1 conv). Dependent
# hyperparameters are used to turn the h_filter_size and
# h_stride parameters--which only take values of ints--into the necessary
# parameters for the spatially separable convolution.
def conv_spatial_separable(h_num_filters, h_filter_size, h_stride):
    h_filter_size_1 = co.DependentHyperparameter(lambda size: [1, size],
                                                 {'size': h_filter_size})
    h_filter_size_2 = co.DependentHyperparameter(lambda size: [size, 1],
                                                 {'size': h_filter_size})
    h_stride_1 = co.DependentHyperparameter(lambda stride: [1, stride],
                                            {'stride': h_stride})
    h_stride_2 = co.DependentHyperparameter(lambda stride: [stride, 1],
                                            {'stride': h_stride})
    return mo.siso_sequential([
        conv2d(h_num_filters, h_filter_size_1, h_stride_1, D([1]), D([True])),
        batch_normalization(),
        relu(),
        conv2d(h_num_filters, h_filter_size_2, h_stride_2, D([1]), D([True]))
    ])


# This is a simple convenience function used to wrap a conv module with relu
# and batch norm
def wrap_relu_batch_norm(conv):
    return mo.siso_sequential([relu(), conv, batch_normalization()])


# This function applies some of the convolution specific logic that is not
# specified in the paper but is used in the code base. Essentially, it transforms
# the convolution into a bottleneck layer. The dependent hyperparameters are
# being used in two ways below. One way is to calculate the bottleneck filter
# size, and the other (in conjunction with an Optional subsitituion module) to
# decide whether to apply the bottleneck at all.
def apply_conv_op(main_op, op_name, h_num_filter):
    if op_name is 's_sep3':
        reduced_filter_size = co.DependentHyperparameter(
            lambda size: int(3 * size / 8), {'size': h_num_filter},
            name='reduced_filter_size')
    else:
        reduced_filter_size = co.DependentHyperparameter(
            lambda size: int(size / 4), {'size': h_num_filter},
            name='reduced_filter_size')

    bottleneck_too_thin = co.DependentHyperparameter(
        lambda size: size < 1, {'size': reduced_filter_size},
        name='bottleneck_too_thin')

    bottleneck_filters = co.DependentHyperparameter(
        lambda reduced_filter_size, num_filter: num_filter if reduced_filter_size < 1 else reduced_filter_size,
        {
            'reduced_filter_size': reduced_filter_size,
            'num_filter': h_num_filter
        },
        name='bottleneck_filters')

    return mo.siso_sequential([
        mo.siso_optional(
            lambda: wrap_relu_batch_norm(
                conv2d(reduced_filter_size, D([1]), D([1]), D([1]), D([True]))
            ), bottleneck_too_thin
        ),
        wrap_relu_batch_norm(main_op(bottleneck_filters)),
        mo.siso_optional(
            lambda: wrap_relu_batch_norm(
                conv2d(h_num_filter, D([1]), D([1]), D([1]), D([True])),
            ), bottleneck_too_thin
        )
    ])


# This module is used to encode the logic for 1x1 convolution and no op
# operations. It essentially adds a stride by using a 1x1 convolution to the
# no op layer if needed.
def check_stride(h_stride, h_num_filter, h_op_name):
    need_stride = co.DependentHyperparameter(
        lambda name, stride: int(name is 'conv1' or stride > 1),
        {
            'name': h_op_name,
            'stride': h_stride
        },
    )
    return mo.siso_or([
        mo.identity,
        lambda: wrap_relu_batch_norm(
            conv2d(h_num_filter, D([1]), h_stride, D([1]), D([True]))
        )
    ], need_stride)


# This module is used to apply the pooling layer specific logic used for
# striding by the amoebanet code.
def check_reduce(main_op, h_stride, h_num_filter):
    need_stride = co.DependentHyperparameter(
        lambda stride: int(stride > 1),
        {'stride': h_stride},
    )
    return mo.siso_sequential([
        main_op,
        mo.siso_optional(
            lambda: mo.siso_sequential([
                conv2d(h_num_filter, D([1]), h_stride, D([1]), D([True])),
                batch_normalization()
            ]), need_stride
        )
    ])


# The regularized evolution paper mentions in the appendix that for the depth
# separable convolutions, they applied them twice whenever selected. This
# is the implementation of that scheme that is used in the amoebanet code.
def stacked_depth_separable_conv(h_filter_size, h_num_filters,
                                 h_depth_multiplier, h_stride):
    return mo.siso_sequential([
        relu(),
        separable_conv2d(h_num_filters, h_filter_size, D([1]), D([1]),
                         h_depth_multiplier, D([True])),
        batch_normalization(),
        relu(),
        separable_conv2d(h_num_filters, h_filter_size, h_stride, D([1]),
                         h_depth_multiplier, D([True])),
        batch_normalization()
    ])


# This is simply an Or substitution module that chooses between the operations
# used in search space 1 and search space 3 in the regularized evolution paper.
def sp1_operation(h_op_name, h_stride, h_filters):
    return mo.siso_or({
        'identity': lambda: check_stride(h_stride, h_filters, h_op_name),
        'd_sep3': lambda: stacked_depth_separable_conv(D([3]), h_filters, D([1]), h_stride),
        'd_sep5': lambda: stacked_depth_separable_conv(D([5]), h_filters, D([1]), h_stride),
        'd_sep7': lambda: stacked_depth_separable_conv(D([7]), h_filters, D([1]), h_stride),
        'avg3': lambda: check_reduce(avg_pool2d(D([3]), D([1])), h_stride, h_filters),
        'max3': lambda: check_reduce(max_pool2d(D([3]), D([1])), h_stride, h_filters),
        'dil3_2': lambda: apply_conv_op(
            lambda filters: conv2d(filters, D([3]), h_stride, D([2]), D([True])),
            'dil3_2', h_filters),
        's_sep7': lambda:apply_conv_op(
            lambda filters: conv_spatial_separable(filters, D([7]), h_stride),
            's_sep7', h_filters),
        }, h_op_name)


# This is simply an Or substitution module that chooses between the operations
# used in search space 2 in the regularized evolution paper.
def sp2_operation(h_op_name, h_stride, h_filters):
    return mo.siso_or({
        'identity': lambda: check_stride(h_stride, h_filters, h_op_name),
        'conv1': lambda: check_stride(h_stride, h_filters, h_op_name),
        'conv3': lambda: apply_conv_op(
            lambda filters: conv2d(filters, D([3]), h_stride, D([1]), D([True])),
            'conv3', h_stride),
        'd_sep3': lambda: stacked_depth_separable_conv(D([3]), h_filters, D([1]), h_stride),
        'd_sep5': lambda: stacked_depth_separable_conv(D([5]), h_filters, D([1]), h_stride),
        'd_sep7': lambda: stacked_depth_separable_conv(D([7]), h_filters, D([1]), h_stride),
        'avg2': lambda: check_reduce(avg_pool2d(D([2]), D([1])), h_stride, h_filters),
        'avg3': lambda: check_reduce(avg_pool2d(D([3]), D([1])), h_stride, h_filters),
        'max2': lambda: check_reduce(max_pool2d(D([2]), D([1])), h_stride, h_filters),
        'max3': lambda: check_reduce(max_pool2d(D([3]), D([1])), h_stride, h_filters),
        'dil3_2': lambda: apply_conv_op(
            lambda filters: conv2d(filters, D([3]), h_stride, D([2]), D([True])),
            'dil3_2', h_stride),
        'dil4_2': lambda: apply_conv_op(
            conv2d(h_filters, D([3]), h_stride, D([4]), D([True])),
            'dil3_4', h_stride),
        'dil6_2': lambda: apply_conv_op(
            lambda filters: conv2d(filters, D([3]), h_stride, D([6]), D([True])),
            'dil3_6', h_stride),
        's_sep3': lambda:apply_conv_op(
            lambda filters: conv_spatial_separable(filters, D([3]), h_stride),
            's_sep3', h_stride),
        's_sep7': lambda:apply_conv_op(
            lambda filters: conv_spatial_separable(filters, D([7]), h_stride),
            's_sep7', h_stride),
        }, h_op_name)


# This module takes in a specifiable number of inputs, uses 1x1 convolutions to
# make the number filters match, and then adds the inputs together
def mi_add(num_terms, name=None):

    def compile_fn(di, dh):
        (_, _, _, min_channels) = di['In0'].get_shape().as_list()
        for i in range(num_terms):
            (_, _, _, channels) = di['In' + str(i)].get_shape().as_list()
            min_channels = channels if channels < min_channels else min_channels

        W_init_fn = cnn2d.kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)
        w_dict = {}
        b_dict = {}
        for i in range(num_terms):
            (_, _, _, channels) = di['In' + str(i)].get_shape().as_list()
            if channels != min_channels:
                w_dict[i] = tf.Variable(
                    W_init_fn([1, 1, channels, min_channels]))
                b_dict[i] = tf.Variable(b_init_fn([min_channels]))

        def fn(di):
            trans = []
            for i in range(num_terms):
                if i in w_dict:
                    trans.append(
                        tf.nn.bias_add(
                            tf.nn.conv2d(di['In' + str(i)], w_dict[i],
                                         [1, 1, 1, 1], 'SAME'), b_dict[i]))
                else:
                    trans.append(di['In' + str(i)])
            total_sum = trans[0]
            for i in range(1, num_terms):
                total_sum = tf.add(total_sum, trans[i])
            return {'Out': total_sum}

        return fn

    return TFM('MultiInputAdd' if name is None else name, {}, compile_fn,
               ['In' + str(i) for i in range(num_terms)], ['Out']).get_io()


# This module applies a sp1_operation to two inputs, and adds them together
def sp1_combine(h_op1_name, h_op2_name, h_stride_1, h_stride_2, h_filters):
    op1 = sp1_operation(h_op1_name, h_stride_1, h_filters)
    op2 = sp1_operation(h_op2_name, h_stride_2, h_filters)
    ops = [op1, op2]
    add_inputs, add_outputs = mi_add(len(ops))

    i_inputs = OrderedDict()
    for i in range(len(ops)):
        i_inputs['In' + str(i)] = ops[i][0]['In']
        add_inputs['In' + str(i)].connect(ops[i][1]['Out'])
    return (i_inputs, add_outputs)


# This module is a substitution module that selects a module from a list of
# input modules.
def selector(h_selection, available, scope=None):

    def substitution_fn(selection):
        ins, outs = available[selection]
        if len(ins) > 1:
            values = list(ins.values())
            ins = OrderedDict({'In': values[0]})
        return ins, outs

    return mo.substitution_module('Selector', {'selection': h_selection},
                                  substitution_fn, ['In'], ['Out'], scope)


# This module takes in a list of cell inputs, a list of hyperparameters that
# will specify which cell inputs are used, and then adds up the unused inputs.
# It is a substitution module that results in only the unused inputs being
# connected to the module that adds them together. Because we don't know how
# inputs will end up being connected to the substitution module (since this is
# a value determined by the list of selection hyperparameters), we
# need to use a trick where we create the substitution module using a dummy
# function. Then, our true substitution function, which will be called once all
# of the selection hyperparameters are specified and we can determine the number
# of inputs to the substitution/add module, will directly modify the inputs
# dictionary for the module according to how many inputs we need. It will also
# connect all of the relevant input modules to the created add module.
# Finally, we replace the dummy substitution module for our module with the true
# substitution function.
def add_unused(h_selections, available, normal, name=None, scope=None):

    def dummy_func(**dummy_hs):
        return mo.identity()

    module = mo.SubstitutionModule('AddUnused', {
        'selection' + str(i): h_selections[i] for i in range(len(h_selections))
    }, dummy_func, ['In'], ['Out'], scope)

    def substitution_fn(**selections):
        selected = [False] * len(selections)
        for k in selections:
            selected[selections[k]] = True

        # Disconnect first two factorized reductions if they are not used
        for i in range(2):
            if selected[i] and not normal:
                available[i][1]['Out'].module.inputs['In'].disconnect()

        unused = [
            available[i] for i in range(len(available)) if not selected[i]
        ]
        ins, outs = mi_add(len(unused), name=name)

        module.inputs = {}
        for i in range(len(unused)):
            module._register_input('In' + str(i))
            unused[i][1]['Out'].connect(ins['In' + str(i)])

        return ins, outs

    module._substitution_fn = substitution_fn
    return module.get_io()


# This is a module used to pad and shift the input. This is used as part of the
# factorized reduction operation in the amoebanet code.
def pad_and_shift():

    def compile_fn(di, dh):
        pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]

        def fn(di):
            return {'Out': tf.pad(di['In'], pad_arr)[:, 1:, 1:, :]}

        return fn

    return htf.siso_tensorflow_module('Pad', compile_fn, {})


# This is a module used to concatenate two inputs along the channel dimension.
# This is used as part of the factorized reduction operation in the amoebanet
# code.
def concat(axis):

    def compile_fn(di, dh):

        def fn(di):
            return {'Out': tf.concat(values=[di['In0'], di['In1']], axis=axis)}

        return fn

    return TFM('Concat', {}, compile_fn, ['In0', 'In1'], ['Out']).get_io()


# This operation is used to reduce the size of the input, either by striding
# ir reducing the number of filters, without losing information. It is specified
# in the amoebanet code. The reduction of the number of filters is fairly
# straightforward. The striding is done by splitting the input along two paths,
# both of which are strided with half the number of final filters, and then
# concatenated along the channel dimensions. We use the `is_stride_2` dependent
# hyperparameter to decide whether we need to do the striding operations.
def factorized_reduction(h_num_filters, h_stride):
    is_stride_2 = co.DependentHyperparameter(lambda stride: int(stride == 2),
                                             {'stride': h_stride})
    reduced_filters = co.DependentHyperparameter(lambda filters: filters / 2,
                                                 {'filters': h_num_filters})
    ins, outs = mo.identity()
    path1_ins, path1_outs = mo.siso_sequential([
        avg_pool2d(D([1]), h_stride),
        conv2d(reduced_filters, D([1]), D([1]), D([1]), D([True]))
    ])
    path2_ins, path2_outs = mo.siso_sequential([
        pad_and_shift(),
        avg_pool2d(D([1]), h_stride),
        conv2d(reduced_filters, D([1]), D([1]), D([1]), D([True]))
    ])
    c_ins, c_outs = concat(3)
    path1_outs['Out'].connect(c_ins['In0'])
    outs['Out'].connect(path1_ins['In'])

    path2_outs['Out'].connect(c_ins['In1'])
    outs['Out'].connect(path2_ins['In'])

    return mo.siso_or([
        lambda: mo.siso_sequential([
            conv2d(h_num_filters, D([1]), h_stride, D([1]), D([True])),
            batch_normalization()
        ]),
        lambda: mo.siso_sequential([
            (ins, c_outs),
            batch_normalization()
        ])
    ], is_stride_2, name='factor')


# A module that outlines the basic cell structure in the transferable
# architectures paper. It uses a hyperparameter sharer to get the same
# architecture specification across cells. It takes in an array called
# `available` which contains the previous two outputs. Then, it builds the cell
# according to the process described at the top of this tutorial.
def basic_cell(available, h_sharer, h_filters, C=5, normal=True):
    assert len(available) == 2
    i_inputs = OrderedDict([('In' + str(idx), available[idx][1]['Out'])
                            for idx in range(len(available))])
    selections = []
    for i in range(C):
        if normal:
            h_op0 = h_sharer.get('h_norm_op0_' + str(i))
            h_op1 = h_sharer.get('h_norm_op1_' + str(i))
            h_in0_pos = h_sharer.get('h_norm_in0_pos_' + str(i))
            h_in1_pos = h_sharer.get('h_norm_in1_pos_' + str(i))
        else:
            h_op0 = h_sharer.get('h_red_op0_' + str(i))
            h_op1 = h_sharer.get('h_red_op1_' + str(i))
            h_in0_pos = h_sharer.get('h_red_in0_pos_' + str(i))
            h_in1_pos = h_sharer.get('h_red_in1_pos_' + str(i))

        selections.append(h_in0_pos)
        selections.append(h_in1_pos)
        sel0_inputs, sel0_outputs = selector(h_in0_pos, available[:])
        sel1_inputs, sel1_outputs = selector(h_in1_pos, available[:])

        def is_stride_2(pos):
            return 1 if normal or pos > 1 else 2

        h_stride_0 = co.DependentHyperparameter(
            is_stride_2, {'pos': h_in0_pos}, name='h_stride_0')
        h_stride_1 = co.DependentHyperparameter(
            is_stride_2, {'pos': h_in1_pos}, name='h_stride_1')
        comb_inputs, comb_outputs = sp1_combine(h_op0, h_op1, h_stride_0,
                                                h_stride_1, h_filters)
        sel0_outputs['Out'].connect(comb_inputs['In0'])
        sel1_outputs['Out'].connect(comb_inputs['In1'])
        available.append((comb_inputs, comb_outputs))

    if not normal:
        available[0] = mo.siso_sequential(
            [available[0],
             factorized_reduction(h_filters, D([2]))])
        available[1] = mo.siso_sequential(
            [available[1],
             factorized_reduction(h_filters, D([2]))])

    add_inputs, add_outputs = add_unused(
        selections, available[:], normal,
        'NORMAL_CELL' if normal else 'REDUCTION_CELL')
    return i_inputs, add_outputs


# This module creates a number of repeated cells (both reduction and normal)
# according to the overall architecture parameter specification. It is a
# substitution module, so it will not exist once the architecture is fully
# specified.
def ss_repeat(input_layers,
              h_N,
              h_sharer,
              h_filters,
              C,
              num_ov_repeat,
              num_classes,
              scope=None):

    def substitution_fn(N):
        assert N > 0

        h_iter = h_filters
        for i in range(num_ov_repeat):
            for j in range(N):
                available = input_layers[-2:]
                norm_cell = basic_cell(
                    available, h_sharer, h_iter, C=C, normal=True)
                input_layers.append(norm_cell)
            if i < num_ov_repeat - 1:
                h_iter = co.DependentHyperparameter(
                    lambda filters: filters * 2, {'filters': h_iter},
                    name='h_iter')
                available = input_layers[-2:]
                red_cell = basic_cell(
                    available, h_sharer, h_iter, C=C, normal=False)
                input_layers[-1] = mo.siso_sequential(
                    [input_layers[-1],
                     factorized_reduction(h_iter, D([2]))])
                input_layers.append(red_cell)

        return mo.siso_sequential(
            [input_layers[-1],
             global_pool2d(),
             fc_layer(D([num_classes]))])

    return mo.substitution_module('SS_repeat', {'N': h_N}, substitution_fn,
                                  ['In0', 'In1'], ['Out'], scope)


# This function creates a search space using operations from search spaces 1
# and 3 from the regularized evolution paper
def get_search_space_small(num_classes, C):
    co.Scope.reset_default_scope()
    C = 5
    h_N = D([3])
    h_F = D([24])
    h_sharer = hp.HyperparameterSharer()
    for i in range(C):
        h_sharer.register(
            'h_norm_op0_' + str(i),
            lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 'dil3_2', 's_sep7'], name='Mutatable'))
        h_sharer.register(
            'h_norm_op1_' + str(i),
            lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 'dil3_2', 's_sep7'], name='Mutatable'))
        h_sharer.register(
            'h_norm_in0_pos_' + str(i),
            lambda i=i: D(list(range(2 + i)), name='Mutatable'))
        h_sharer.register(
            'h_norm_in1_pos_' + str(i),
            lambda i=i: D(list(range(2 + i)), name='Mutatable'))
    for i in range(C):
        h_sharer.register(
            'h_red_op0_' + str(i),
            lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 's_sep7'], name='Mutatable'))
        h_sharer.register(
            'h_red_op1_' + str(i),
            lambda: D(['identity', 'd_sep3', 'd_sep5', 'd_sep7', 'avg3', 'max3', 's_sep7'], name='Mutatable'))
        h_sharer.register(
            'h_red_in0_pos_' + str(i),
            lambda i=i: D(list(range(2 + i)), name='Mutatable'))
        h_sharer.register(
            'h_red_in1_pos_' + str(i),
            lambda i=i: D(list(range(2 + i)), name='Mutatable'))
    i_inputs, i_outputs = conv2d(h_F, D([1]), D([1]), D([1]), D([True]))
    o_inputs, o_outputs = ss_repeat([(i_inputs, i_outputs),
                                     (i_inputs, i_outputs)], h_N, h_sharer, h_F,
                                    C, 3, num_classes)
    return mo.siso_sequential(
        [mo.identity(), (i_inputs, o_outputs),
         mo.identity()])


# This function creates search space 1 from the regularized evolution paper
def get_search_space_1(num_classes):
    return get_search_space_small(num_classes, 5)


# This function creates search space 3 from the regularized evolution paper
def get_search_space_3(num_classes):
    return get_search_space_small(num_classes, 15)


# This function creates search space 2 from the regularized evolution paper
def get_search_space_2(num_classes):
    co.Scope.reset_default_scope()
    C = 5
    h_N = D([3])
    h_F = D([24])
    h_sharer = hp.HyperparameterSharer()
    for i in range(C):
        h_sharer.register(
            'h_norm_op0_' + str(i),
            lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3', 'min2', 'max2', 'max3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6'], name='Mutatable'))
        h_sharer.register(
            'h_norm_op1_' + str(i),
            lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3', 'min2', 'max2', 'max3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6'], name='Mutatable'))
        h_sharer.register(
            'h_in0_pos_' + str(i),
            lambda i=i: D(list(range(2 + i)), name='Mutatable'))
        h_sharer.register(
            'h_in1_pos_' + str(i),
            lambda i=i: D(list(range(2 + i)), name='Mutatable'))
    for i in range(C):
        if i == 0:
            h_sharer.register('h_red_op0_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
            'min2', 'max2', 'max3', 's_sep3' 's_sep7'], name='Mutatable'))
            h_sharer.register('h_red_op1_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
            'min2', 'max2', 'max3', 's_sep3' 's_sep7'], name='Mutatable'))
        else:
            h_sharer.register(
                'h_red_op0_' + str(i),
                lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3', 'min2', 'max2', 'max3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6'], name='Mutatable'))
            h_sharer.register('h_red_op1_' + str(i), lambda: D(['identity', 'conv1', 'conv3', 'd_sep3', 'd_sep5', 'd_sep7', 'avg2', 'avg3',
            'min2', 'max2', 'max3', 'dil3', 'dil5', 'dil7', 's_sep3' 's_sep7', 'dil3_2', 'dil3_4', 'dil3_6'], name='Mutatable'))
        h_sharer.register(
            'h_red_in0_pos_' + str(i),
            lambda i=i: D(list(range(2 + i)), name='Mutatable'))
        h_sharer.register(
            'h_red_in1_pos_' + str(i),
            lambda i=i: D(list(range(2 + i)), name='Mutatable'))
    i_inputs, i_outputs = conv2d(h_F, D([1]), D([1]), D([1]), D([True]))
    o_inputs, o_outputs = ss_repeat([(i_inputs, i_outputs),
                                     (i_inputs, i_outputs)], h_N, h_sharer, h_F,
                                    C, 3, num_classes)
    return mo.siso_sequential(
        [mo.identity(), (i_inputs, o_outputs),
         mo.identity()])


# A simple search space factory to create the appropriate search space after
# getting the search space name.
class SSFZoph17(mo.SearchSpaceFactory):

    def __init__(self, search_space, num_classes):
        if search_space == 'sp1':
            search_space_fn = get_search_space_1
        elif search_space == 'sp2':
            search_space_fn = get_search_space_2
        elif search_space == 'sp3':
            search_space_fn = get_search_space_3
        mo.SearchSpaceFactory.__init__(self,
                                       lambda: search_space_fn(num_classes))
