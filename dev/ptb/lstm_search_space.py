import deep_architect.core as co
import deep_architect.hyperparameters as hp
import deep_architect.utils as ut
import deep_architect.helpers.tensorflow as htf
import deep_architect.modules as mo
import tensorflow as tf
import numpy as np
import random

TFM = htf.TFModule
D = hp.Discrete

def siso_tfm(name, compile_fn, name_to_h={}, scope=None):
    return htf.TFModule(name, name_to_h, compile_fn,
            ['In'], ['Out'], scope).get_io()

# Learned embeddings module
def embeddings(h_embedding_size, vocab_size):
    def cfn(di, dh):
        embedding = tf.get_variable("embedding", [vocab_size, dh['embedding_size']], dtype=tf.float32)
        def fn(di):
            return {'Out':tf.nn.embedding_lookup(embedding, di['In'])}
        return fn

    return siso_tfm('Embeddings', cfn,{
        'embedding_size': h_embedding_size
        })

# Creates an lstm cell (Note: Not a module)
def lstm_cell(hidden_size, keep_prob):
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,
            forget_bias=0.0, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell

# Creates an Multi layer LSTM module
def multi_rnn_cell(h_hidden_size, h_keep_prob, h_num_layers, batch_size, num_steps):
    def cfn(di, dh):
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(dh['hidden_size'],
            dh['keep_prob']) for _ in range(dh['num_layers'])], state_is_tuple=True)
        def fn(di):
            initial_state = cell.zero_state(batch_size, tf.float32)
            state = initial_state
            outputs = []
            with tf.variable_scope("RNN"):
              for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(di['In'][:, time_step, :], state)
                outputs.append(cell_output)
            output = tf.reshape(tf.concat(outputs, 1), [-1, dh['hidden_size']])
            return {'Out': output}
        return fn
    return siso_tfm('MultiRNNCell', cfn, {
        'hidden_size': h_hidden_size,
        'num_layers': h_num_layers,
        'keep_prob': h_keep_prob
        })


def softmax(h_hidden_size, batch_size, num_steps, vocab_size):
    def cfn(di, dh):
        softmax_w = tf.get_variable("softmax_w", [dh['hidden_size'], vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        def fn(di):
            logits = tf.nn.xw_plus_b(di['In'], softmax_w, softmax_b)
            # Reshape logits to be a 3-D tensor for sequence loss
            logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])
            return {'Out': logits}
        return fn
    return siso_tfm('Softmax', cfn, {
        'hidden_size': h_hidden_size
        })

def hyperparameters_fn():
    return {
        'optimizer_type' : D([ 'adam' ]),
        'lr_start' : D( np.logspace(-1, -4, num=16) ),
        'stop_patience' : D([ 512 ]),
        'lr_end' : D([ 1e-6 ]),
        # 'max_num_epochs' : D([ 100 ]),
        # 'angle_delta' : D([ 0, 5, 10, 15, 20, 25, 30, 35 ]),
        # 'scale_delta' : D([ 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35 ]),
        'weight_decay_coeff' : D([ 0.0, 1e-6, 1e-5, 1e-4 ]),
        # 'scale_delta' : D([ 0.0, 0.05, 0.1, 0.2 ]),
        # 'angle_delta' : D([ 0, 5, 10, 15, 20 ])
        }

# Creates a basic search space using an LSTM for the Penn Treebank dataset
def ptb_search_space(batch_size, num_steps, vocab_size):
    h_sharer = hp.HyperparameterSharer()
    h_embedding_size = D([300])
    h_sharer.register('h_hidden_size', lambda:D([200, 650, 1500]))
    h_num_layers= D([2])
    h_keep_prob = D([1, .5, .35])

    inputs, outputs = mo.siso_sequential([
        embeddings(h_embedding_size, vocab_size),
        multi_rnn_cell(h_sharer.get('h_hidden_size'), h_keep_prob, h_num_layers, batch_size, num_steps),
        softmax(h_sharer.get('h_hidden_size'), batch_size, num_steps, vocab_size)
        ])
    return inputs, outputs, hyperparameters_fn()


