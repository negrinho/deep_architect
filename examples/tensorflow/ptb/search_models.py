import tensorflow as tf
import darch.core as co
import darch.hyperparameters as hps
import darch.utils as ut
import darch.searchers as se
import darch.tensorflow_helpers as tf_helpers
from utils import vocabulary
from tensorflow.python.ops import math_ops, rnn_cell_impl, array_ops


def SISOTFM(name, name_to_h, compile_fn, scope=None):
    return tf_helpers.TFModule(name, name_to_h, compile_fn,
            ['In'], ['Out'], scope)
TFM = SISOTFM
D = hps.Discrete

EMBEDDING_SIZE = 40


def emb_fn(embedding_size, start=-1., end=1.):
    def f(In):
        embeddings = tf.Variable(tf.random_uniform([vocabulary, embedding_size], start, end), name='embeddings')
        embedded_x = tf.nn.embedding_lookup(embeddings, In)
        return {'Out': embedded_x}
    return f


def rnn_fn(num_units, forget_bias=1.0, activation=None,
           forget_activation=None, output_activation=None,
           reuse=None):
    return lambda In: {'Out': tf.nn.dynamic_rnn(
        CustomBasicLSTMCell(num_units, forget_bias,
                            activation, forget_activation,
                            output_activation, reuse),
        inputs=In,
        dtype=tf.float32,
        time_major=False
    )[0]}


class CustomBasicLSTMCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, forget_bias=1.0,
                 activation=None, forget_activation=None,
                 output_activation=None, reuse=None):
        rnn_cell_impl.RNNCell.__init__(self, _reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh
        self._forget_activation = forget_activation or math_ops.sigmoid
        self._output_activation = output_activation or math_ops.sigmoid

    @property
    def state_size(self):
        return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        c, h = state

        concat = rnn_cell_impl._linear([inputs, h], 4 * self._num_units, True)

        i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

        new_c = (
            c * self._forget_activation(f + self._forget_bias) + self._forget_activation(i) * self._activation(j))
        new_h = self._activation(new_c) * self._output_activation(o)

        new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)

        return new_h, new_state


def affine_fn(size):
    return lambda In: {'Out': tf.layers.dense(In, size)}


def get_darch_models(x):
    xs = [
        TFM('Embedding', {
            'embedding_size': D([ EMBEDDING_SIZE ])
        }, emb_fn),
        TFM('CustomLSTM', {
            'num_units': D([ 20, 40, 60 ]),
            'forget_activation': D([ math_ops.sigmoid, math_ops.tanh ]),
            'activation': D([ math_ops.log_sigmoid, math_ops.sigmoid, math_ops.tanh])
        }, rnn_fn),
        TFM('Affine', {
            'size': D([ vocabulary ])
        }, affine_fn)
    ]

    ut.connect_sequentially(xs)
    In, Out = xs[0].inputs['In'], xs[-1].outputs['Out']

    models_outputs = []
    for i in range(5):
        with tf.variable_scope('model' + str(i)):
            se.random_specify([Out])
            co.forward({In : x})
            logits =  Out.val
            models_outputs.append(logits)
    return models_outputs
