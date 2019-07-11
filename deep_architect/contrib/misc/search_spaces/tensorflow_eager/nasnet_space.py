from __future__ import absolute_import
from builtins import str
from builtins import range

import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils

import deep_architect.core as co
import deep_architect.hyperparameters as hp
import deep_architect.helpers.tensorflow_eager_support as htfe
import deep_architect.modules as mo
from deep_architect.contrib.deep_learning_backend.tensorflow_keras_ops import (
    relu, batch_normalization, conv2d, separable_conv2d, avg_pool2d, max_pool2d,
    min_pool2d, fc_layer, global_pool2d, dropout, add, flatten)

from deep_architect.hyperparameters import Discrete as D

global_vars = {}
hp_sharer = hp.HyperparameterSharer()


def cell(input_fn, node_fn, combine_fn, unused_combine_fn, num_nodes,
         hyperparameters):

    def substitution_fn(**dh):
        c_ins, c_outs = input_fn()
        nodes = [c_outs['Out0'], c_outs['Out1']]
        used_node = [False] * (num_nodes + 2)
        for i in range(num_nodes):
            # Get indices of hidden states to be combined
            idx0 = dh[str(i) + '_0']
            idx1 = dh[str(i) + '_1']

            # Transform hidden states
            h0 = node_fn(idx0, i, 0)
            h1 = node_fn(idx1, i, 1)
            h0[0]['In'].connect(nodes[idx0])
            h1[0]['In'].connect(nodes[idx1])
            used_node[idx0] = used_node[idx1] = True

            # Combine hidden states
            h = combine_fn()
            h[0]['In0'].connect(h0[1]['Out'])
            h[0]['In1'].connect(h1[1]['Out'])

            nodes.append(h[1]['Out'])

        ins, outs = unused_combine_fn(sum(not used for used in used_node))
        input_id = 0
        for ix, node in enumerate(nodes):
            if not used_node[ix]:
                ins['In' + str(input_id)].connect(node)
                input_id += 1
        return c_ins, outs

    name_to_hyperp = {
        '%d_%d' % (i // 2, i % 2): hyperparameters[i]
        for i in range(len(hyperparameters))
    }
    return mo.substitution_module('Cell', name_to_hyperp, substitution_fn,
                                  ['In0', 'In1'], ['Out'], None)


def SP1_ops(name=None, reduction=False):
    if reduction:
        # Dilated convolution can't be done with strides
        ops = [
            'none', 'depth_sep3', 'depth_sep5', 'depth_sep7', 'avg3', 'max3',
            '1x7_7x1'
        ]
    else:
        ops = [
            'none', 'depth_sep3', 'depth_sep5', 'depth_sep7', 'avg3', 'max3',
            'dilated_3x3_rate_2', '1x7_7x1'
        ]
    return D(ops, name=name)


def wrap_relu_batch_norm(op):
    return mo.siso_sequential([relu(), op, batch_normalization()])


def conv_op(filters, filter_size, stride, dilation_rate, spatial_separable):
    if spatial_separable:
        return mo.siso_sequential([
            conv2d(D([filters]), D([[1, filter_size]]), D([[1, stride]])),
            batch_normalization(),
            relu(),
            conv2d(D([filters]), D([[filter_size, 1]]), D([[stride, 1]])),
        ])
    else:
        return conv2d(D([filters]), D([filter_size]), D([stride]),
                      D([dilation_rate]))


def full_conv_op(filters, filter_size, stride, dilation_rate,
                 spatial_separable):
    # Add bottleneck layer according to
    # https://github.com/tensorflow/tpu/blob/master/models/official/amoeba_net/network_utils.py
    if filter_size == 3 and spatial_separable:
        reduced_filter_size = int(3 * filters / 8)
    else:
        reduced_filter_size = int(filters / 4)
    if reduced_filter_size < 1:
        return wrap_relu_batch_norm(
            conv_op(filters, filter_size, stride, dilation_rate,
                    spatial_separable))
    else:
        return mo.siso_sequential([
            wrap_relu_batch_norm(conv2d(D([reduced_filter_size]), D([1]))),
            wrap_relu_batch_norm(
                conv_op(reduced_filter_size, filter_size, stride, dilation_rate,
                        spatial_separable)),
            wrap_relu_batch_norm(conv2d(D([filters]), D([1])))
        ])


def check_filters(filters, stride=1):

    def compile_fn(di, dh):
        num_filters = di['In'].shape[-1].value
        if num_filters != filters or stride > 1:
            conv = tf.keras.layers.Conv2D(filters,
                                          1,
                                          strides=stride,
                                          padding='SAME')
        else:
            conv = None

        def forward_fn(di, is_training=True):
            return {'Out': di['In'] if conv is None else conv(di['In'])}

        return forward_fn

    return htfe.siso_tensorflow_eager_module('CheckFilters', compile_fn, {})


def pool_op(filters, filter_size, stride, pool_type):
    if pool_type == 'avg':
        pool = avg_pool2d(D([filter_size]), D([stride]))
    elif pool_type == 'max':
        pool = max_pool2d(D([filter_size]), D([stride]))
    else:
        pool = min_pool2d(D([filter_size]), D([stride]))

    return mo.siso_sequential([pool, check_filters(filters)])


def separable_conv_op(filters, filter_size, stride):
    return mo.siso_sequential([
        wrap_relu_batch_norm(
            separable_conv2d(D([filters]), D([filter_size]), D([stride]))),
        wrap_relu_batch_norm(
            separable_conv2d(D([filters]), D([filter_size]), D([1])))
    ])


def apply_drop_path(x, keep_prob):
    batch_size = tf.shape(x)[0]
    noise_shape = [batch_size, 1, 1, 1]
    keep_prob = tf.cast(keep_prob, dtype=x.dtype)
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape, dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)
    return tf.div(x, keep_prob) * binary_tensor


class DropPath(tf.keras.layers.Layer):

    def __init__(self, init_keep_prob, cell_ratio, total_steps):
        super(DropPath, self).__init__()
        self.init_keep_prob = 1 - cell_ratio * (1 - init_keep_prob)
        self.compute_prob = tf.keras.layers.Lambda(lambda x: 1 - x * (
            1 - self.init_keep_prob))
        self.total_steps = total_steps

    def build(self, shape):
        super(DropPath, self).build(shape)
        self.progress = tf.divide(
            tf.cast(tf.train.get_or_create_global_step(), tf.float32),
            self.total_steps)

    def call(self, x, training=None):
        if training is None:
            training = True
        keep_prob = self.compute_prob(self.progress)
        return tf_utils.smart_cond(
            training, lambda: apply_drop_path(x, keep_prob), lambda: x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'init_keep_prob': self.init_keep_prob,
            'total_steps': self.total_steps
        }
        base_config = super(DropPath, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def drop_path(cell_ratio):

    def compile_fn(di, dh):
        drop_path_layer = DropPath(dh['keep_prob'], cell_ratio, di['In1'])

        def forward_fn(di, is_training=True):
            return {'Out': drop_path_layer(di['In0'])}

        return forward_fn

    return htfe.TensorflowEagerModule(
        'DropPath', {
            'keep_prob': hp_sharer.get('drop_path_keep_prob')
        }, compile_fn, ['In0', 'In1'], ['Out']).get_io()


class MISOIdentity(co.Module):
    """Module passes the input to the output without changes.

    Args:
        scope (deep_architect.core.Scope, optional): Scope in which the module will be
            registered. If none is given, uses the default scope.
        name (str, optional): Name used to derive an unique name for the
            module. If none is given, uses the class name to derive
            the name.
    """

    def __init__(self, scope=None, name=None):
        co.Module.__init__(self, scope, name)
        self._register_input("In0")
        self._register_input("In1")
        self._register_output("Out")

    def _compile(self):
        pass

    def _forward(self):
        self.outputs['Out'].val = self.inputs['In0'].val


def miso_optional(fn, h_opt):

    def substitution_fn(opt):
        return fn() if opt else MISOIdentity().get_io()

    return mo.substitution_module("MISOOptional", {'opt': h_opt},
                                  substitution_fn, ['In0', 'In1'], ['Out'],
                                  scope=None)


def intermediate_node_fn(reduction, input_id, node_id, op_num, filters,
                         cell_ratio, cell_ops):
    stride = 2 if reduction and input_id < 2 else 1
    h_is_not_none = co.DependentHyperparameter(lambda op: op != 'none', {
        'op':
        cell_ops[node_id * 2 + op_num]
    })

    op_in, op_out = mo.siso_or(
        {
            'none':
            lambda: check_filters(filters, stride),
            'conv1':
            lambda: wrap_relu_batch_norm(
                conv2d(D([filters]), D([1]), h_stride=D([stride]))),
            'conv3':
            lambda: full_conv_op(filters, 3, stride, 1, False),
            'depth_sep3':
            lambda: separable_conv_op(filters, 3, stride),
            'depth_sep5':
            lambda: separable_conv_op(filters, 5, stride),
            'depth_sep7':
            lambda: separable_conv_op(filters, 7, stride),
            'dilated_3x3_rate_2':
            lambda: full_conv_op(filters, 3, stride, 2, False),
            'dilated_3x3_rate_4':
            lambda: full_conv_op(filters, 3, stride, 4, False),
            'dilated_3x3_rate_6':
            lambda: full_conv_op(filters, 3, stride, 6, False),
            '1x3_3x1':
            lambda: full_conv_op(filters, 3, stride, 1, True),
            '1x7_7x1':
            lambda: full_conv_op(filters, 7, stride, 1, True),
            'avg2':
            lambda: pool_op(filters, 2, stride, 'avg'),
            'avg3':
            lambda: pool_op(filters, 3, stride, 'avg'),
            'max2':
            lambda: pool_op(filters, 2, stride, 'max'),
            'max3':
            lambda: pool_op(filters, 3, stride, 'max'),
            'min2':
            lambda: pool_op(filters, 2, stride, 'min')
        }, cell_ops[node_id * 2 + op_num])
    drop_in, drop_out = miso_optional(lambda: drop_path(cell_ratio),
                                      h_is_not_none)
    drop_in['In0'].connect(op_out['Out'])
    drop_in['In1'].connect(global_vars['total_steps'])
    return op_in, drop_out


def concat(num_ins):

    def compile_fn(di, dh):
        concat_op = tf.keras.layers.Concatenate(axis=3) if num_ins > 1 else None

        def forward_fn(di, is_training=True):
            return {
                'Out':
                concat_op([di[input_name] for input_name in di])
                if concat_op else di['In0']
            }

        return forward_fn

    return htfe.TensorflowEagerModule('Concat', {}, compile_fn,
                                      ['In' + str(i) for i in range(num_ins)],
                                      ['Out']).get_io()


def combine_unused(num_ins):
    inputs = [mo.identity() for _ in range(num_ins)]
    factorized = [maybe_factorized_reduction() for _ in range(num_ins)]
    concat_ins, concat_outs = concat(num_ins)
    di = {}
    last_in, last_out = inputs[-1]
    for i in range(num_ins):
        f_in, f_out = inputs[i]
        factorized[i][0]['In0'].connect(f_out['Out'])
        factorized[i][0]['In1'].connect(last_out['Out'])
        di['In' + str(i)] = f_in['In']
        concat_ins['In' + str(i)].connect(factorized[i][1]['Out'])

    return di, concat_outs


def maybe_factorized_reduction(add_relu=False):

    def compile_fn(di, dh):
        _, _, height, channels = di['In0'].shape.as_list()
        _, _, final_height, final_channels = di['In1'].shape.as_list()
        if add_relu:
            relu = tf.keras.layers.ReLU()
        if height == final_height and channels == final_channels:
            pass
        elif height == final_height:
            conv = tf.keras.layers.Conv2D(final_channels, 1)
            bn = tf.keras.layers.BatchNormalization()
        else:
            avg_pool = tf.keras.layers.AveragePooling2D(1, 2, padding='VALID')
            conv1 = tf.keras.layers.Conv2D(int(final_channels / 2), 1)
            conv2 = tf.keras.layers.Conv2D(int(final_channels / 2), 1)
            bn = tf.keras.layers.BatchNormalization()
            pad = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)))
            slice_layer = tf.keras.layers.Lambda(lambda x: x[:, 1:, 1:, :])
            concat = tf.keras.layers.Concatenate(axis=3)

        def forward_fn(di, is_training=True):
            inp = relu(di['In0']) if add_relu else di['In0']
            if height == final_height and channels == final_channels:
                out = inp
            elif height == final_height:
                out = conv(inp)
                out = bn(out)
            else:
                path1 = avg_pool(inp)
                path1 = conv1(path1)
                path2 = slice_layer(pad(inp))
                path2 = avg_pool(path2)
                path2 = conv2(path2)
                out = concat([path1, path2])
                out = bn(out)
            return {'Out': out}

        return forward_fn

    return htfe.TensorflowEagerModule('MaybeFactorizedReduction', {},
                                      compile_fn, ['In0', 'In1'],
                                      ['Out']).get_io()


def cell_input_fn(filters):
    prev_input = mo.identity()
    cur_input = wrap_relu_batch_norm(conv2d(D([filters]), D([1])))
    transformed_prev_input = maybe_factorized_reduction(add_relu=True)
    transformed_prev_input[0]['In0'].connect(prev_input[1]['Out'])
    transformed_prev_input[0]['In1'].connect(cur_input[1]['Out'])
    return {
        'In0': prev_input[0]['In'],
        'In1': cur_input[0]['In']
    }, {
        'Out0': transformed_prev_input[1]['Out'],
        'Out1': cur_input[1]['Out']
    }


def create_cell_generator(num_nodes, reduction):
    prefix = 'reduction' if reduction else 'normal'
    cell_ops = [
        SP1_ops('%s_op_%d_%d' % (prefix, i // 2, i % 2), reduction)
        for i in range(2 * num_nodes)
    ]
    connection_hparams = [
        D(list(range(i // 2 + 2)), name='%s_in_%d_%d' % (prefix, i // 2, i % 2))
        for i in range(2 * num_nodes)
    ]

    def generate(filters, cell_ratio):
        return cell(
            lambda: cell_input_fn(filters), lambda in_id, node_id, op_num:
            intermediate_node_fn(reduction, in_id, node_id, op_num, filters,
                                 cell_ratio, cell_ops), lambda: add(2),
            combine_unused, num_nodes, connection_hparams)

    return generate


def stem(filters):
    return mo.siso_sequential(
        [conv2d(D([filters]), D([3])),
         batch_normalization()])


def global_convolution(h_num_filters):

    def compile_fn(di, dh):
        _, h, w, _ = di['In'].shape.as_list()
        conv = tf.keras.layers.Conv2D(dh['num_filters'], [h, w],
                                      use_bias=False,
                                      padding='VALID')

        def forward_fn(di, is_training=True):
            return {'Out': conv(di['In'])}

        return forward_fn

    return htfe.siso_tensorflow_eager_module('GlobalConv2D', compile_fn, {
        'num_filters': h_num_filters,
    })


def aux_logits():
    return mo.siso_sequential([
        relu(),
        avg_pool2d(D([5]), D([3]), D(['VALID'])),
        conv2d(D([128]), D([1])),
        batch_normalization(),
        relu(),
        global_convolution(D([768])),
        batch_normalization(),
        relu(),
        flatten(),
        fc_layer(D([10]))
    ])


def generate_search_space(num_nodes_per_cell, num_normal_cells,
                          num_reduction_cells, init_filters, stem_multiplier):
    global global_vars, hp_sharer
    global_vars = {}
    hp_sharer = hp.HyperparameterSharer()
    hp_sharer.register(
        'drop_path_keep_prob', lambda: D([.7], name='drop_path_keep_prob'))
    stem_in, stem_out = stem(int(init_filters * stem_multiplier))
    total_steps_in, total_steps_out = mo.identity()
    global_vars['total_steps'] = total_steps_out['Out']
    normal_cell_fn = create_cell_generator(num_nodes_per_cell, False)
    reduction_cell_fn = create_cell_generator(num_nodes_per_cell, True)

    total_cells = num_normal_cells + num_reduction_cells
    has_reduction = [False] * num_normal_cells
    for i in range(num_reduction_cells):
        has_reduction[int(
            float(i + 1) / (num_reduction_cells + 1) * num_normal_cells)] = True

    inputs = [stem_out, stem_out]
    filters = init_filters
    aux_loss_idx = int(
        float(num_reduction_cells) /
        (num_reduction_cells + 1) * num_normal_cells) - 1

    outs = {}
    cells_created = 0.0
    for i in range(num_normal_cells):
        if has_reduction[i]:
            filters *= 2
            connect_new_cell(
                reduction_cell_fn(filters, (cells_created + 1) / total_cells),
                inputs)
            cells_created += 1.0
        connect_new_cell(
            normal_cell_fn(filters, (cells_created + 1) / total_cells), inputs)
        cells_created += 1.0
        if i == aux_loss_idx:
            aux_in, aux_out = aux_logits()
            aux_in['In'].connect(inputs[-1]['Out'])
            outs['Out0'] = aux_out['Out']
    _, final_out = mo.siso_sequential([(None, inputs[-1]),
                                       relu(),
                                       global_pool2d(),
                                       dropout(D([1.0])),
                                       fc_layer(D([10]))])
    outs['Out1'] = final_out['Out']
    return {'In0': stem_in['In'], 'In1': total_steps_in['In']}, outs


def connect_new_cell(cell, inputs):
    cell[0]['In0'].connect(inputs[-2]['Out'])
    cell[0]['In1'].connect(inputs[-1]['Out'])
    inputs.append(cell[1])


class SSF_NasnetA(mo.SearchSpaceFactory):

    def __init__(self):
        mo.SearchSpaceFactory.__init__(
            self, lambda: generate_search_space(5, 18, 2, 36, 3.0))
