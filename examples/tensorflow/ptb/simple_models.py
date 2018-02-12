import tensorflow as tf
import darch.core as co
import darch.hyperparameters as hps
import darch.utils as ut
import darch.searchers as se
import darch.tensorflow_helpers as tf_helpers
from utils import vocabulary

def SISOTFM(name, name_to_h, compile_fn, scope=None):
    return tf_helpers.TFModule(name, name_to_h, compile_fn,
            ['In'], ['Out'], scope)
TFM = SISOTFM
D = hps.Discrete

EMBEDDING_SIZE = 40
LSTM_SIZE = 20


def emb_fn(embedding_size, start=-1., end=1.):
    def f(In):
        embeddings = tf.Variable(tf.random_uniform([vocabulary, embedding_size], start, end), name='embeddings')
        embedded_x = tf.nn.embedding_lookup(embeddings, In)
        return {'Out': embedded_x}
    return f


def lstm_fn(lstm_size):
    return lambda In: {'Out': tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.BasicLSTMCell(lstm_size),
        inputs=In,
        dtype=tf.float32,
        time_major=False
    )[0]}


def affine_fn(size):
    return lambda In: {'Out': tf.layers.dense(In, size)}


def get_ref_model(x):
    embedded_x = emb_fn(x, EMBEDDING_SIZE)(x)

    rnn_out = lstm_fn(LSTM_SIZE)(embedded_x)

    dropped_output = rnn_out  # TODO put dropout here
    output = affine_fn(vocabulary)(dropped_output)

    return output


def get_darch_model(x):
    xs = [
        TFM('Embedding', {
            'embedding_size': D([ EMBEDDING_SIZE ])
        }, emb_fn),
        TFM('LSTM', {
            'lstm_size': D([ LSTM_SIZE ])
        }, lstm_fn),
        TFM('Affine', {
            'size': D([ vocabulary ])
        }, affine_fn)
    ]

    ut.connect_sequentially(xs)
    In, Out = xs[0].inputs['In'], xs[-1].outputs['Out']
    se.random_specify([Out])
    co.forward({In : x})
    logits =  Out.val
    return logits
