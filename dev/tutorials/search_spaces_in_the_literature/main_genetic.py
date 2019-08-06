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


def conv2d_with_relu(filters, kernel_size):
    return hke.siso_keras_module_from_keras_layer_fn(lambda: Conv2D(
        filters, kernel_size, padding='same', activation='relu', use_bias=False
    ), {},
                                                     name="Conv2D")


def batch_normalization():
    return hke.siso_keras_module_from_keras_layer_fn(BatchNormalization, {})


def conv2d_cell(filters, kernel_size):
    return mo.siso_sequential(
        [conv2d_with_relu(filters, kernel_size),
         batch_normalization()])


# do one with compile functions.

# def max_pooling():
#     return hke.siso_keras_module_from_keras_layer_fn()


#
def combine_with_sum(num_inputs):

    m = co.Module(name="SumCombiner")
    m._register(["In%d" % i for i in range(num_inputs)], ['Out'], {})
    return m.get_io()


def conv_stage(filters, kernel_size, num_nodes):

    def substitution_fn(dh):
        print dh
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

            node_ids_using_any = set()
            node_ids_used = set()
            for i, lst in iteritems(node_id_to_lst):
                for d in lst:
                    if d['use']:
                        node_ids_using_any.add(d["node_id"])
                        node_ids_used.add(d["in_node_id"])

            node_ids_to_ignore = set([
                i for i in range(num_nodes)
                if i not in node_ids_using_any and i not in node_ids_used
            ])

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
                    else:
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
            # all zeros encoded.
            return conv2d_cell(filters, kernel_size)

    name_to_hyperp = {
        ut.json_object_to_json_string({
            "node_id": i,
            "in_node_id": j
        }): D([0, 1]) for i in range(num_nodes) for j in range(i)
    }
    return mo.substitution_module("ConvStage",
                                  substitution_fn,
                                  name_to_hyperp, ["In"], ["Out"],
                                  scope=None)


(inputs, outputs
) = mo.SearchSpaceFactory(lambda: conv_stage(32, 3, 8)).get_search_space()
random_specify(outputs)
# for h in co.unassigned_independent_hyperparameter_iterator(outputs):
#     h.assign_value(1)

vi.draw_graph(outputs, draw_module_hyperparameter_info=False)

# inputs_val = Input((32, 32, 3))
# co.forward({inputs["In"]: inputs_val})
# outputs_val = outputs["Out"].val

# model = Model(inputs=inputs_val, outputs=outputs_val)
# model.summary()

### NOTE: these are done.

# TODO: finish the model