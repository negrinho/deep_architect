from six import iteritems, itervalues
import tensorflow as tf
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Conv2D,
                                     DepthwiseConv2D, SeparableConv2D,
                                     MaxPooling2D, AveragePooling2D,
                                     Concatenate)
import deep_architect.core as co
import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.utils as ut
from deep_architect.searchers.common import random_specify
import dev.helpers.tfeager as htfe
from dev.deep_learning_backend.tfe_ops import batch_normalization, relu

# import deep_architect.helpers.keras_support as hke

# import deep_architect.visualization as vi

D = hp.Discrete

# what the best way to do about this lol
# from keras.models import Model

###############################################################################
# Base motifs (level=1, motif_num=6)
###############################################################################
# def batch_normalization():
#     return htfe.siso_tfeager_module_from_tensorflow_op_fn(BatchNormalization, {})


def conv2d(filters, kernel_size, stride=1, activation_fn='linear'):
    return htfe.siso_tfeager_module_from_tensorflow_op_fn(
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
    return htfe.siso_tfeager_module_from_tensorflow_op_fn(
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
    return htfe.siso_tfeager_module_from_tensorflow_op_fn(
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
    return htfe.siso_tfeager_module_from_tensorflow_op_fn(
        lambda: MaxPooling2D(pool_size, strides=stride, padding='same'), {},
        name="MaxPooling2D_%dx%d" % (pool_size, pool_size))


# average-pooling
def average_pooling(pool_size, stride=1):
    return htfe.siso_tfeager_module_from_tensorflow_op_fn(
        lambda: MaxPooling2D(pool_size, strides=stride, padding='same'), {},
        name="AveragePooling2D_%dx%d" % (pool_size, pool_size))


###############################################################################
# Motif and other functions (level>=1)
###############################################################################


def global_average_pooling():
    return htfe.siso_tfeager_module_from_tensorflow_op_fn(
        GlobalAveragePooling2D, {})


def dense(units):
    return htfe.siso_tfeager_module_from_tensorflow_op_fn(
        lambda: Dense(units), {})


# depthwise concatenation
def combine_with_concat(num_inputs, channels):
    return mo.siso_sequential([concat(num_inputs), conv2d_cell(channels, 1)])


def concat(num_inputs):
    input_names = ["In%d" % i for i in range(num_inputs)]

    def compile_fn(di, dh):
        if num_inputs > 1:
            concat = Concatenate()

        def forward_fn(di, isTraining=False):
            return {
                "Out":
                concat([v for name, v in iteritems(di)])
                if num_inputs > 1 else di['In0']
            }

        return forward_fn

    return htfe.TFEModule("ConcatCombiner", {}, compile_fn, input_names,
                          ['Out']).get_io()


def create_motif_substitution_module(level,
                                     name_to_hyperp,
                                     submotifs,
                                     num_nodes,
                                     num_channel,
                                     top_level=False):
    """Create a substitution module for a motif, based on (shared) dictionary of
    hyperparameters and list of previous submotifs
    Args:
        level: motif at level >= 2
        name_to_hyperp: dictionary of hyperparameters, with each hyperparmeters
            being an edge labeled with integer k indicating that edge operation
            is submotif_k
        submotifs[() -> substitution_module]: list of submotifs from previous level
            note that it is a function for delayed evaluation, so each submotif
            only connects to this motif and not other motif (although sharing the
            same configurations via name_to_hyperp)
        num_nodes: number of nodes in graph at this level
    Returns:
        Substitution Module of this motif at this level
    """
    assert num_nodes >= 1

    def substitution_fn(**dh):
        # print("level=",level, "dh=", dh)
        node_id_to_node_ids_used = {i: [] for i in range(1, num_nodes)}
        for name, v in iteritems(dh):
            if v:  # include edges if D != 0
                d = ut.json_string_to_json_object(name)
                i, j = d["out_node_id"], d["in_node_id"]
                node_ids_used = node_id_to_node_ids_used[i]
                node_ids_used.append((j, v))
        for i in range(1, num_nodes):
            node_id_to_node_ids_used[i] = sorted(node_id_to_node_ids_used[i])

        (inputs, outputs) = mo.identity()
        node_id_to_outputs = [outputs]
        in_inputs = inputs
        for i in range(1, num_nodes):
            node_ids_used = node_id_to_node_ids_used[i]
            num_edges = len(node_ids_used)

            outputs_lst = []
            for (j, k) in node_ids_used:
                # print("level=%d, k=%d" % (level, k))
                # print(submotifs)
                (submotif_fn,
                 submotif_dict) = submotifs[k - 1]  # note 1 <= k <= num_motifs
                inputs, outputs = submotif_fn(submotif_dict)
                j_outputs = node_id_to_outputs[j]
                inputs["In"].connect(j_outputs["Out"])
                outputs_lst.append(outputs)

            # if necessary, concatenate the results going into a node
            if num_edges > 1:
                c_inputs, c_outputs = combine_with_concat(
                    num_edges, num_channel)
                for idx, outputs in enumerate(outputs_lst):
                    c_inputs["In%d" % idx].connect(outputs["Out"])
            else:
                c_outputs = outputs_lst[0]
            if level == 2 and not top_level:
                conv_ins, conv_outs = conv2d_cell(num_channel, 1)
                c_outputs['Out'].connect(conv_ins['In'])
                c_outputs = conv_outs
            node_id_to_outputs.append(c_outputs)

        out_outputs = node_id_to_outputs[-1]
        return in_inputs, out_outputs

    return mo.substitution_module("Motif_level%d" % level,
                                  name_to_hyperp,
                                  substitution_fn, ["In"], ["Out"],
                                  scope=None)


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
        # motif_hyperps.update({
        #     ut.json_object_to_json_string({
        #         "in_node_id": j,
        #         "out_node_id": i,
        #         "level": level
        #     }): D(list(range(prev_num_motifs + 1)))
        #     for i in range(1, num_nodes) for j in range(i)
        # })
        # prev_num_motifs = num_motifs
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


def motifs(motif_hyperp, level=1):
    """Recursively define hierarchial representations
    Args:
        motif_hyperp: a dictionary of hyperparameters for all motifs
        num_channel: number of channel for convolutions operations
    Returns
        list of motif modules at that level
    """
    if level == 1:  # base motifs
        return [
            lambda num_channel: conv2d_cell(num_channel, 1
                                           ),  # 1x1 conv of C channels
            lambda num_channel: depthwise_conv2d_cell(3),  # 3x3 depthwise conv
            lambda num_channel: separable_conv2d_cell(
                num_channel, 3),  # 3x3 sep conv of C channels
            lambda num_channel: max_pooling(3),  # 3x3 max pool
            lambda num_channel: average_pooling(3),  # 3x3 avg pool
            lambda num_channel: mo.identity()
        ]
    else:
        motifs_lst = []
        # prev_motifs = motifs(motif_hyperp, num_channel, level - 1)
        (motif_dicts, num_nodes) = motif_hyperp[level]
        for motif_dict in motif_dicts:
            # use closure here so that each motif is created at different graph,
            # although still sharing the same configurations
            motif_fn = lambda num_channel, motif_dict=motif_dict: create_motif_substitution_module(
                level,
                motif_dict,
                motifs(motif_hyperp, level - 1),
                num_nodes,
                num_channel,
                top_level=len(motif_dicts) == 1)
            motifs_lst.append(motif_fn)
        return motifs_lst


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
        # if level == 1:
        #     # if in_node == -1 or out_node == -1 or out_node <= in_node:
        #     #     raise ValueError(
        #     #         'Bad values for motif creation- in: %d out: %d level: %d' %
        #     #         (in_node, out_node, level))
        #     return mo.siso_or(
        #         [
        #             lambda: conv2d_cell(num_channels, 1
        #                                ),  # 1x1 conv of C channels
        #             lambda: depthwise_conv2d_cell(3),  # 3x3 depthwise conv
        #             lambda: separable_conv2d_cell(
        #                 num_channels, 3),  # 3x3 sep conv of C channels
        #             lambda: max_pooling(3),  # 3x3 max pool
        #             lambda: average_pooling(3),  # 3x3 avg pool
        #             lambda: mo.identity()
        #         ],
        #         D([dh['operation']]))
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
        # if level < len(motif_info) + 1:
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
    motif_info = {
        2: {
            'num_nodes': 11,
            'num_motifs': 1
        }
    }  # originally 11, scale down otherwise out of memory

    motif_hyperp = create_motif_hyperp(motif_info)
    # motif_fn = motifs(motif_hyperp, level=2)[0]
    motif_fn = lambda c: create_motif(motif_hyperp[2][0], motif_hyperp,
                                      motif_info, 2, c)
    return mo.siso_sequential([
        conv2d_cell(c0, kernel_size),  # or conv2d_cell here?
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
        }  # original 5, scaled down otherwise out of memory
    }
    motif_hyperp = create_motif_hyperp(motif_info)
    # motif_fn, _ = motifs(motif_hyperp, level=3)[0]
    motif_fn = lambda c: create_motif(motif_hyperp[3][0], motif_hyperp,
                                      motif_info, 3, c)
    return mo.siso_sequential([
        conv2d_cell(c0, kernel_size),  # or conv2d_cell here?
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


# # (inputs, outputs) = mo.SearchSpaceFactory(
# #     lambda: flat_search_space(10)).get_search_space()
# (inputs, outputs) = mo.SearchSpaceFactory(
#     lambda: hierarchical_search_space(10)).get_search_space()
# hyper_lst = random_specify(outputs.values())
# # for h in co.unassigned_independent_hyperparameter_iterator(outputs.values()):
# #     h.assign_value(1)
# # vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)

# # (inputs, outputs) = mo.SearchSpaceFactory(
# #     lambda: hierarchical_search_space(10)).get_search_space()

# # vi.draw_graph_evolution(outputs.values(), hyper_lst, "tmp/", draw_module_hyperparameter_info=True, draw_hyperparameters=False)

# inputs_val = Input((32, 32, 3))
# co.forward({inputs["In"]: inputs_val})
# outputs_val = outputs["Out"].val

# model = Model(inputs=inputs_val, outputs=outputs_val)
# model.summary()
