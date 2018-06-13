"""
Search space from Efficient Neural Architecture Search (Pham'17)
"""
import tensorflow as tf
from collections import OrderedDict

import darch.helpers.tensorflow as htf
import darch.modules as mo
from darch.contrib.useful.search_spaces.tensorflow.common import D, siso_tfm
from .common_ops import wrap_relu_batch_norm, global_pool_and_logits, avg_pool
from darch.contrib.useful.search_spaces.tensorflow import cnn2d

TFM = htf.TFModule
const_fn = lambda c: lambda shape: tf.constant(c, shape=shape)

class WeightSharer():
    def __init__(self):
        self.name_to_weight = OrderedDict()
    
    def get(self, name, weight_fn):
        if name not in self.name_to_weight:
            self.name_to_weight[name] = weight_fn()
            print name
        return self.name_to_weight[name]
        

def conv2D(filter_size, name, weight_sharer):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        W_init_fn = cnn2d.kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        W_fn = lambda: tf.Variable( W_init_fn( [filter_size, filter_size, channels, channels] ) )
        b_fn = lambda: tf.Variable( b_init_fn( [channels] ) )
        W = weight_sharer.get(name + '_conv_weight_' + str(filter_size), W_fn)
        b = weight_sharer.get(name + '_conv_bias_' + str(filter_size), b_fn)
        def fn(di):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(di['In'], W, [1, 1, 1, 1], 'SAME'), b)}
        return fn

    return siso_tfm('Conv2D', cfn, {})

def conv2D_depth_separable(filter_size, name, weight_sharer):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        W_init_fn = cnn2d.kaiming2015delving_initializer_conv()
        b_init_fn = const_fn(0.0)

        W_depth_fn = lambda: tf.Variable( W_init_fn( [filter_size, filter_size, channels, 1] ) )
        W_point_fn = lambda: tf.Variable( W_init_fn( [1, 1, channels, channels] ) )
        b_fn = lambda: tf.Variable( b_init_fn( [channels] ) )

        W_depth = weight_sharer.get(name + '_dsep_depth_' + str(filter_size), W_depth_fn)
        W_point = weight_sharer.get(name + '_dsep_point_' + str(filter_size), W_point_fn)
        b = weight_sharer.get(name + '_dsep_bias_' + str(filter_size), b_fn)
        def fn(di):
            return {'Out' : tf.nn.bias_add(
                tf.nn.separable_conv2d(di['In'], W_depth, W_point, [1, 1, 1, 1], 'SAME', [1, 1]), b)}
        return fn

    return siso_tfm('Conv2DSeparable', cfn, {})

def enas_space(h_num_layers, fn_first, fn_repeats, input_names, output_names, weight_sharer, scope=None):
    def sub_fn(num_layers):
        assert num_layers > 0
        inputs, outputs = fn_first()
        temp_outputs = OrderedDict(outputs)
        for i in range(1, num_layers + 1):
            inputs, temp_outputs = fn_repeats(inputs, temp_outputs, i, weight_sharer)
        return inputs, OrderedDict({'Out': temp_outputs['Out' + str(len(temp_outputs) - 1)]})
    return mo.substitution_module('ENASModule', {'num_layers': h_num_layers}, 
                                  sub_fn, input_names, output_names, scope)

# Take in array of boolean hyperparams, concatenate layers corresponding to true
# to form skip connections
def concatenate_skip_layers(h_connects, weight_sharer):
    def cfn(di, dh):
        inputs = [di['In' + str(i)] for i in range(len(dh)) if dh['select_' + str(i)]]
        inputs.append(di['In' + str(len(dh))])

        # if sum([dh['select_' + str(i)] for i in range(len(di))]) > 0:
        #     inputs = [di['In' + str(i)] for i in range(len(di)) if dh['select_' + str(i)]]
        # else:
        #     inputs = di.values()
        # with tf.variable_scope("training_flag", reuse=tf.AUTO_REUSE):
        p_var = weight_sharer.get('training_pl', lambda: tf.placeholder(tf.bool))
        (_, _, _, channels) = inputs[0].get_shape().as_list()

        def fn(di):
            out = tf.add_n(inputs)
            # if len(inputs) > 1:
            #     stacked = tf.concat(inputs, 3)
            #     dim_red_op = tf.layers.Conv2D(channels, (channels, channels),
            #         (1, 1), use_bias=1, padding='SAME')
            #     stacked = dim_red_op(stacked)
            out = tf.layers.batch_normalization(out, training=p_var)
            return {'Out' : out}
            # else:
            #     return {'Out' : inputs[0]}

        return fn, {p_var : 1}, {p_var : 0}
    return TFM('SkipConcat', 
               {'select_' + str(i) : h_connects[i] for i in range(len(h_connects))},
               cfn, ['In' + str(i) for i in range(len(h_connects) + 1)], ['Out']).get_io()

def enas_op(h_op_name, name, weight_sharer):
    return mo.siso_or({
        'conv3': lambda: wrap_relu_batch_norm(conv2D(3, name, weight_sharer), weight_sharer),
        'conv5': lambda: wrap_relu_batch_norm(conv2D(5, name, weight_sharer), weight_sharer),
        'dsep_conv3': lambda: wrap_relu_batch_norm(conv2D_depth_separable(3, name, weight_sharer), weight_sharer),
        'dsep_conv5': lambda: wrap_relu_batch_norm(conv2D_depth_separable(5, name, weight_sharer), weight_sharer),
        'avg_pool': lambda: avg_pool(D([3]), D([1])),
        'max_pool': lambda: cnn2d.max_pool2d(D([3]), D([1]))
    }, h_op_name)

def enas_repeat_fn(inputs, outputs, layer_id, weight_sharer):
    h_enas_op = D(['conv3', 'conv5', 'dsep_conv3', 'dsep_conv5', 'avg_pool', 'max_pool'], name='op_' + str(layer_id))
    op_inputs, op_outputs = enas_op(h_enas_op, 'op_' + str(layer_id), weight_sharer)
    outputs[outputs.keys()[-1]].connect(op_inputs['In'])

    h_connects = [D([True, False], name='skip_'+str(idx)+'_'+str(layer_id)) for idx in range(layer_id - 1)]
    skip_inputs, skip_outputs = concatenate_skip_layers(h_connects, weight_sharer)

    for i in range(len(h_connects)):
        outputs[outputs.keys()[i]].connect(skip_inputs['In' + str(i)])
    # for i, out_name in enumerate(outputs):
    #     outputs[out_name].connect(skip_inputs['In' + str(i)])
    op_outputs['Out'].connect(skip_inputs['In' + str(len(h_connects))])


    # skip_outputs['Out'].connect(op_inputs['In'])
    outputs['Out' + str(len(outputs))] = skip_outputs['Out']
    return inputs, outputs
    
def fc_softmax(num_classes, weight_sharer):
    def cfn(di, dh):
        (_, _, _, channels) = di['In'].get_shape().as_list()
        W_init_fn = cnn2d.kaiming2015delving_initializer_conv()

        W_fn = lambda: tf.get_variable('final_fc', shape=(channels, num_classes))
        W = weight_sharer.get('softmax_weight', W_fn)
        def fn(di):
            logits = tf.matmul(tf.squeeze(di['In'], axis=[1, 2]), W)
            return {'Out' : tf.nn.softmax(logits)}
        return fn
    return siso_tfm('fc_softmax', cfn, {}) 

def get_enas_search_space(num_classes, out_filters, weight_sharer):
    h_N = D([5], name='num_layers')
    return mo.siso_sequential([
        enas_space(
            h_N, 
            lambda: cnn2d.conv2d(D([out_filters]), D([3]), D([1]), D([False])), 
            enas_repeat_fn, ['In'], ['Out'], weight_sharer),
        global_pool_and_logits(),
        fc_softmax(num_classes, weight_sharer),
        ])

class SSFEnasnet(mo.SearchSpaceFactory):
    def __init__(self, num_classes, out_filters):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes
        self.weight_sharer = WeightSharer()
        self.out_filters = out_filters

    def _get_search_space(self):
        inputs, outputs = get_enas_search_space(self.num_classes, self.out_filters, self.weight_sharer)
        return inputs, outputs, {}