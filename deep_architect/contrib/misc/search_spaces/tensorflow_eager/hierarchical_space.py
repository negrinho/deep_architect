from six import iteritems, itervalues
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Conv2D,
                                     DepthwiseConv2D, SeparableConv2D,
                                     MaxPooling2D, AveragePooling2D,
                                     Concatenate)
import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.utils as ut
import deep_architect.helpers.tensorflow_eager_support as htfe
from deep_architect.contrib.deep_learning_backend.tfe_ops import batch_normalization

D = hp.Discrete


def conv2d(filters, kernel_size, stride=1, activation_fn='linear'):
    return htfe.siso_tensorflow_eager_module_from_tensorflow_op_fn(
        lambda: Conv2D(filters,
                       kernel_size,
                       strides=stride,
                       padding='same',
                       activation='relu',
                       use_bias=False), {},
        name="Conv2D_%dx%d" % (kernel_size, kernel_size))


# convolution of filters=C channels (followed by relu and batch norm)
def conv2d_cell(filters, kernel_size, stride=1):
    return mo.siso_sequential([
        batch_normalization(),
        conv2d(filters, kernel_size, stride, 'relu'),
    ])


def depthwise_conv2d(kernel_size, stride=1, activation_fn='linear'):
    return htfe.siso_tensorflow_eager_module_from_tensorflow_op_fn(
        lambda: DepthwiseConv2D(kernel_size,
                                strides=stride,
                                padding='same',
                                activation=activation_fn,
                                use_bias=False), {},
        name="DepthwiseConv2D_%dx%d" % (kernel_size, kernel_size))


# depthwise convolution (followed by relu and batch norm)
def depthwise_conv2d_cell(kernel_size, stride=1):
    return mo.siso_sequential(
        [depthwise_conv2d(kernel_size, stride, 'relu'),
         batch_normalization()])


def separable_conv2d(filters, kernel_size, stride=1, activation_fn='linear'):
    return htfe.siso_tensorflow_eager_module_from_tensorflow_op_fn(
        lambda: SeparableConv2D(filters,
                                kernel_size,
                                strides=stride,
                                padding='same',
                                activation=activation_fn,
                                use_bias=False), {},
        name="SeparableConv2D_%dx%d" % (kernel_size, kernel_size))


# separable convolution of filters=C channels (followed by relu and batch norm)
def separable_conv2d_cell(filters, kernel_size, stride=1):
    return mo.siso_sequential([
        separable_conv2d(filters, kernel_size, stride, 'relu'),
        batch_normalization()
    ])


# max-pooling
def max_pooling(pool_size, stride=1):
    return htfe.siso_tensorflow_eager_module_from_tensorflow_op_fn(
        lambda: MaxPooling2D(pool_size, strides=stride, padding='same'), {},
        name="MaxPooling2D_%dx%d" % (pool_size, pool_size))


# average-pooling
def average_pooling(pool_size, stride=1):
    return htfe.siso_tensorflow_eager_module_from_tensorflow_op_fn(
        lambda: AveragePooling2D(pool_size, strides=stride, padding='same'), {},
        name="AveragePooling2D_%dx%d" % (pool_size, pool_size))


###############################################################################
# Motif and other functions (level>=1)
###############################################################################


def global_average_pooling():
    return htfe.siso_tensorflow_eager_module_from_tensorflow_op_fn(
        GlobalAveragePooling2D, {})


def dense(units):
    return htfe.siso_tensorflow_eager_module_from_tensorflow_op_fn(
        lambda: Dense(units), {})


# depthwise concatenation
def combine_with_concat(num_inputs, channels):
    return mo.siso_sequential([concat(num_inputs), conv2d_cell(channels, 1)])


def concat(num_inputs):
    input_names = ["In%d" % i for i in range(num_inputs)]

    def compile_fn(di, dh):
        if num_inputs > 1:
            concat = Concatenate()

        def forward_fn(di, is_training=False):
            return {
                "Out":
                concat([v for name, v in iteritems(di)])
                if num_inputs > 1 else di['In0']
            }

        return forward_fn

    return htfe.TensorflowEagerModule("ConcatCombiner", {}, compile_fn,
                                      input_names, ['Out']).get_io()


def create_motif_hyperp(motif_info):
    """
    Create dictionary of hyperparameters for each motif for each level in
    hierarchial representations. The dictionaries can be shared across a graph G,
    e.g. if edge e and edge v have the same motif k, then e and v will share
    dictionary of hyperp k
    Args:
        Motif_info: list containing motif information, assumed to be in the
        following format:
        [(level_k, ({'num_nodes': x, 'num_motifs': y}), ... (level_n, ...)),
                    num_nodes_level_k))
        ]
    Returns:
        a dictionary of hyperparameters in the following format:
        {level_num: [dict_motif_1, dict_motif_2,...,dict_motif_y]}
        where dict_motif_k = {edge ij: D([0,1,...,w])},
        w is num_motif at previous level, 2 <= level_num
    """
    motif_hyperp = dict()
    prev_num_motifs = 6  # 6 primitive operations
    # motif_hyperps = {}

    for (level, info) in sorted(motif_info.items()):
        num_nodes, num_motifs = info['num_nodes'], info['num_motifs']
        motif_dicts = []
        for m in range(num_motifs):
            # chain nodes: node i needs to be immediately followed by node i-1, and
            # the hyperp of this edge can never be 0 (representing non-edge)
            motif_dict = {
                ut.json_object_to_json_string({
                    "out_node_id": i,
                    "in_node_id": i - 1
                }): D(list(range(1, prev_num_motifs + 1)))
                for i in range(1, num_nodes)
            }

            # non-chain nodes: node i and j (j != i-1) can be non-edge
            motif_dict.update({
                ut.json_object_to_json_string({
                    "out_node_id": i,
                    "in_node_id": j,
                }): D(list(range(prev_num_motifs + 1)))
                for i in range(1, num_nodes) for j in range(i - 1)
            })

            motif_dicts.append(motif_dict)
        motif_hyperp[level] = motif_dicts
        prev_num_motifs = num_motifs

    return motif_hyperp


def create_motif(
        dh,
        full_hparam_dict,
        motif_info,
        level,
        num_channels,
):
    if level == 1:
        return [
            lambda: conv2d_cell(num_channels, 1),  # 1x1 conv of C channels
            lambda: depthwise_conv2d_cell(3),  # 3x3 depthwise conv
            lambda: separable_conv2d_cell(num_channels, 3
                                         ),  # 3x3 sep conv of C channels
            lambda: max_pooling(3),  # 3x3 max pool
            lambda: average_pooling(3),  # 3x3 avg pool
            lambda: mo.identity()
        ][dh['operation'] - 1]()

    def substitution_fn(**dh):
        num_nodes = motif_info[level]['num_nodes']
        ops = [[] for _ in range(motif_info[level]['num_nodes'])]
        ops[0].append(mo.identity())
        output_ops = []
        for out_node in range(num_nodes):
            for in_node in range(out_node):
                op_id = dh[ut.json_object_to_json_string({
                    "out_node_id": out_node,
                    "in_node_id": in_node,
                })]
                if op_id > 0:
                    if level == 2:
                        next_hparams = {'operation': op_id}
                    else:
                        next_hparams = full_hparam_dict[level - 1][op_id - 1]
                    ops[out_node].append(
                        create_motif(
                            next_hparams,
                            full_hparam_dict,
                            motif_info,
                            level - 1,
                            num_channels,
                        ))
                    ops[out_node][-1][0]['In'].connect(
                        output_ops[in_node][1]['Out'])
            assert (len(ops[out_node]) > 0)
            concat_ins, concat_out = combine_with_concat(
                len(ops[out_node]), num_channels)
            for ix, (ins, outs) in enumerate(ops[out_node]):
                outs['Out'].connect(concat_ins['In%d' % ix])
            output_ops.append((concat_ins, concat_out))
        output = output_ops[-1][1]
        if level == 2:
            conv_ins, conv_outs = conv2d_cell(num_channels, 1)
            output['Out'].connect(conv_ins['In'])
            output = conv_outs
        return ops[0][0][0], output

    return mo.substitution_module('Motif_Level_%d' % level, dh, substitution_fn,
                                  ['In'], ['Out'], None)


# NOTE: description on page 6 of paper
def flat_search_space(num_classes):
    c0 = 64
    kernel_size = 3
    motif_info = {2: {'num_nodes': 11, 'num_motifs': 1}}

    motif_hyperp = create_motif_hyperp(motif_info)
    motif_fn = lambda c: create_motif(motif_hyperp[2][0], motif_hyperp,
                                      motif_info, 2, c)
    return mo.siso_sequential([
        conv2d_cell(c0, kernel_size),
        motif_fn(c0),
        separable_conv2d(c0, kernel_size),
        motif_fn(c0),
        separable_conv2d(2 * c0, kernel_size, stride=2),
        motif_fn(2 * c0),
        separable_conv2d(c0, kernel_size),
        motif_fn(2 * c0),
        separable_conv2d(4 * c0, kernel_size, stride=2),
        motif_fn(4 * c0),
        separable_conv2d(c0, kernel_size),
        motif_fn(4 * c0),
        separable_conv2d(8 * c0, kernel_size, stride=2),
        global_average_pooling(),
        dense(num_classes)
    ])


def hierarchical_search_space(num_classes):

    c0 = 64  # initial number of channel C
    kernel_size = 3
    motif_info = {
        2: {
            'num_nodes': 4,
            'num_motifs': 6
        },
        3: {
            'num_nodes': 5,
            'num_motifs': 1
        }
    }
    motif_hyperp = create_motif_hyperp(motif_info)
    motif_fn = lambda c: create_motif(motif_hyperp[3][0], motif_hyperp,
                                      motif_info, 3, c)
    return mo.siso_sequential([
        conv2d_cell(c0, kernel_size),
        motif_fn(c0),
        separable_conv2d(c0, kernel_size),
        motif_fn(c0),
        separable_conv2d(2 * c0, kernel_size, stride=2),
        motif_fn(2 * c0),
        separable_conv2d(c0, kernel_size),
        motif_fn(2 * c0),
        separable_conv2d(4 * c0, kernel_size, stride=2),
        motif_fn(4 * c0),
        separable_conv2d(c0, kernel_size),
        motif_fn(4 * c0),
        separable_conv2d(8 * c0, kernel_size, stride=2),
        global_average_pooling(),
        dense(num_classes)
    ])
