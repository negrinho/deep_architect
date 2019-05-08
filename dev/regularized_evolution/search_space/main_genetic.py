from six import iteritems, itervalues
from tensorflow.keras.layers import (Flatten, Dense, Dropout, Conv2D, Add,
                                     MaxPooling2D)
import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.utils as ut
# import deep_architect.helpers.keras_support as hke
from dev.helpers import tfeager as htfe
# import deep_architect.visualization as vi

D = hp.Discrete

from dev.deep_learning_backend.tfe_ops import batch_normalization

# from keras.models import Model


def flatten():
    return htfe.siso_tfeager_module_from_tensorflow_op_fn(Flatten, {})


def dense(units):
    return htfe.siso_tfeager_module_from_tensorflow_op_fn(
        lambda: Dense(units), {})


def dropout(rate):
    return htfe.siso_tfeager_module_from_tensorflow_op_fn(
        lambda: Dropout(rate), {})


def conv2d_with_relu(filters, kernel_size):
    return htfe.siso_tfeager_module_from_tensorflow_op_fn(lambda: Conv2D(
        filters, kernel_size, padding='same', activation='relu', use_bias=False
    ), {},
                                                          name="Conv2D")


def conv2d_cell(filters, kernel_size):
    return mo.siso_sequential(
        [conv2d_with_relu(filters, kernel_size),
         batch_normalization()])


def max_pooling(pool_size, stride):
    return htfe.siso_tfeager_module_from_tensorflow_op_fn(lambda: MaxPooling2D(
        pool_size, strides=stride, padding='same'), {},
                                                          name="MaxPooling2D")


def combine_with_sum(num_inputs):

    input_names = ["In%d" % i for i in range(num_inputs)]

    def compile_fn(di, dh):
        add = Add()

        def forward_fn(di, isTraining):
            tensors = [v for name, v in iteritems(di)]
            return {"Out": add(tensors)}

        return forward_fn

    return htfe.TFEModule("SumCombiner", {}, compile_fn, input_names,
                          ['Out']).get_io()


def conv_stage(filters, kernel_size, num_nodes):

    def substitution_fn(**dh):
        any_in_stage = any(v for v in itervalues(dh))

        if any_in_stage:
            node_id_to_lst = {}
            for name, v in iteritems(dh):
                d = ut.json_string_to_json_object(name)
                d['use'] = v
                i = d["node_id"]
                if i not in node_id_to_lst:
                    node_id_to_lst[i] = []
                node_id_to_lst[i].append(d)

            node_ids_using_any = set()  # non-isolated nodes
            node_ids_used = set()  # predecessors of non-isolated nodes
            for i, lst in iteritems(node_id_to_lst):
                for d in lst:
                    if d['use']:
                        node_ids_using_any.add(d["node_id"])
                        node_ids_used.add(d["in_node_id"])
            node_ids_to_ignore = set([
                i for i in range(num_nodes)
                if i not in node_ids_using_any and i not in node_ids_used
            ])  # isolated nodes

            # creating the stage
            (in_inputs, in_outputs) = conv2d_cell(filters, kernel_size)
            node_id_to_outputs = {}
            for i in range(num_nodes):
                if i not in node_ids_to_ignore:
                    if i in node_ids_using_any:
                        in_ids = []
                        for d in node_id_to_lst[i]:
                            if d['use']:
                                in_ids.append(d['in_node_id'])
                        # collecting the inputs for the sum combiner.
                        num_inputs = len(in_ids)
                        if num_inputs > 1:
                            (s_inputs, s_outputs) = combine_with_sum(num_inputs)
                            for idx, j in enumerate(in_ids):
                                j_outputs = node_id_to_outputs[j]
                                s_inputs["In%d" % idx].connect(j_outputs["Out"])
                        else:
                            j = in_ids[0]
                            s_outputs = node_id_to_outputs[j]

                        (n_inputs,
                         n_outputs) = conv2d_cell(filters, kernel_size)
                        n_inputs["In"].connect(s_outputs["Out"])
                    else:  # connect default input node to nodes without predecessors
                        (n_inputs,
                         n_outputs) = conv2d_cell(filters, kernel_size)
                        n_inputs["In"].connect(in_outputs['Out'])
                    node_id_to_outputs[i] = n_outputs

            # final connection to the output node
            in_ids = []
            for i in range(num_nodes):
                if i not in node_ids_to_ignore and i not in node_ids_used:
                    in_ids.append(i)
            num_inputs = len(in_ids)
            if num_inputs > 1:
                (s_inputs, s_outputs) = combine_with_sum(num_inputs)
                for idx, j in enumerate(in_ids):
                    j_outputs = node_id_to_outputs[j]
                    s_inputs["In%d" % idx].connect(j_outputs["Out"])
            else:
                j = in_ids[0]
                s_outputs = node_id_to_outputs[j]

            (out_inputs, out_outputs) = conv2d_cell(filters, kernel_size)
            out_inputs["In"].connect(s_outputs["Out"])
            return (in_inputs, out_outputs)

        else:
            # all zeros encoded, perform convolution once
            return conv2d_cell(filters, kernel_size)

    name_to_hyperp = {
        ut.json_object_to_json_string({
            "node_id": i,
            "in_node_id": j
        }): D([0, 1], name='Mutatable') for i in range(num_nodes)
        for j in range(i)
    }
    return mo.substitution_module("ConvStage",
                                  name_to_hyperp,
                                  substitution_fn, ["In"], ["Out"],
                                  scope=None)


def genetic_cnn(num_classes):

    kernel_size = 3
    pool_stride = 2
    num_filter1, num_filter2, num_filter3 = 8, 16, 32
    num_node1, num_node2, num_node3 = 3, 4, 5
    dense_units = 128
    dropout_rate = 0.5

    return mo.siso_sequential([
        conv_stage(num_filter1, kernel_size, num_node1),
        max_pooling(kernel_size, pool_stride),
        conv_stage(num_filter2, kernel_size, num_node2),
        max_pooling(kernel_size, pool_stride),
        conv_stage(num_filter3, kernel_size, num_node3),
        max_pooling(kernel_size, pool_stride),
        flatten(),
        dense(dense_units),
        dropout(dropout_rate),
        dense(num_classes)
    ])


# (inputs, outputs) = mo.SearchSpaceFactory(lambda: genetic_cnn(10)).get_search_space()
# # random_specify(outputs.values())
# for h in co.unassigned_independent_hyperparameter_iterator(outputs.values()):
#     h.assign_value(1)

# vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)

# inputs_val = Input((32, 32, 3))
# co.forward({inputs["In"]: inputs_val})
# outputs_val = outputs["Out"].val

# model = Model(inputs=inputs_val, outputs=outputs_val)
# model.summary()
