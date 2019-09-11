import deep_architect.core as co
import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.utils as ut
from deep_architect.searchers.common import random_specify
import deep_architect.helpers.keras_support as hke
import deep_architect.visualization as vi

D = hp.Discrete

from keras.layers import Conv2D, BatchNormalization, Input, Add
from keras.models import Model
from six import iteritems, itervalues

# NOTE: add the sum operation. combine these two.
# https://openreview.net/forum?id=BJQRKzbA- ; note this is a useful reference.


def conv2d_with_relu(filters, kernel_size):
    return hke.siso_keras_module_from_keras_layer_fn(
        lambda: Conv2D(filters, kernel_size, padding='same', activation='relu', use_bias=False),
        {}, name="Conv2D")

Conv2D



def batch_normalization():
    return hke.siso_keras_module_from_keras_layer_fn(BatchNormalization, {})


def conv2d_cell(filters, kernel_size):
    return mo.siso_sequential(
        [conv2d_with_relu(filters, kernel_size),
         batch_normalization()])




#
def combine_with_concat(num_inputs):

    m = co.Module(name="ConcatCombiner")
    m._register(["in%d" % i for i in range(num_inputs)], ['out'], {})
    return m.get_io()


def motif_fn(submotif_fn, num_nodes):
    pass


# NOTE: I think that there are dependencies about the motifs.

# NOTE: this is going to be interesting.

# NOTE: I think that this is wrong because it does not sort the mode
#
# # NOTE: I don't think that these are correct because apparently, the
# transformation should be done after the

def average_pooling_3x3():
    hke.siso_keras_module_from_keras_layer_fn()

# 1 × 1 convolution of C channels
# • 3 × 3 depthwise convolution
# • 3 × 3 separable convolution of C channels
# • 3 × 3 max-pooling
# • 3 × 3 average-pooling
# • identity

def base_motif():
    return mo.siso_or([
        conv2d_1x1,
        conv2d_
        lambda: max_pooling(3)
        lambda: average_pooling(3)
    ])


def motif(submotif_fn, num_nodes):
    assert num_nodes >= 1

    def substitution_fn(dh):
        print dh
        node_id_to_node_ids_used = {i: [i - 1] for i in range(1, num_nodes)}
        for name, v in iteritems(dh):
            if v:
                d = ut.json_string_to_json_object(name)
                i = d["node_id"]
                node_ids_used = node_id_to_node_ids_used[i]
                j = d["in_node_id"]
                node_ids_used.append(j)
        for i in range(1, num_nodes):
            node_id_to_node_ids_used[i] = sorted(node_id_to_node_ids_used[i])
        print node_id_to_node_ids_used

        (inputs, outputs) = mo.identity()
        node_id_to_outputs = [outputs]
        in_inputs = inputs
        for i in range(1, num_nodes):
            node_ids_used = node_id_to_node_ids_used[i]
            num_edges = len(node_ids_used)

            outputs_lst = []
            for j in node_ids_used:
                inputs, outputs = submotif_fn()
                j_outputs = node_id_to_outputs[j]
                inputs["in"].connect(j_outputs["out"])
                outputs_lst.append(outputs)

            # if necessary, concatenate the results going into a node
            if num_edges > 1:
                c_inputs, c_outputs = combine_with_concat(num_edges)
                for idx, outputs in enumerate(outputs_lst):
                    c_inputs["in%d" % idx].connect(outputs["out"])
            else:
                c_outputs = outputs_lst[0]
            node_id_to_outputs.append(c_outputs)

        out_outputs = node_id_to_outputs[-1]
        return in_inputs, out_outputs

    name_to_hyperp = {
        ut.json_object_to_json_string({
            "node_id": i,
            "in_node_id": j
        }): D([0, 1]) for i in range(1, num_nodes) for j in range(i - 1)
    }
    return mo.substitution_module(
        "Motif", substitution_fn, name_to_hyperp, ["in"], ["out"], scope=None)


(inputs, outputs) = mo.SearchSpaceFactory(
    lambda: motif(lambda: motif(batch_normalization, 4), 4)).get_search_space()
# (inputs, outputs) = mo.SearchSpaceFactory(
#     lambda: motif(batch_normalization, 4)).get_search_space()
# random_specify(outputs)
for h in co.unassigned_independent_hyperparameter_iterator(outputs):
    h.assign_value(1)

vi.draw_graph(outputs, draw_module_hyperparameter_info=False)

# inputs_val = Input((32, 32, 3))
# co.forward({inputs["in"]: inputs_val})
# outputs_val = outputs["out"].val

# model = Model(inputs=inputs_val, outputs=outputs_val)
# model.summary()

### NOTE: these are done.

# TODO: finish the model