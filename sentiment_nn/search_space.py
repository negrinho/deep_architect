
import sys
sys.path.append('../')

import darch.core as co
import darch.hyperparameters as hp
import darch.helpers.tensorflow as htf
import darch.modules as mo
import darch.searchers as se
import darch.visualization as vi

import tensorflow as tf
from create_sentiment_featuresets import create_feature_sets_and_labels

D = hp.Discrete

def affine(h_m):
    def cfn(di, dh):
        m = dh['m']
        in_dim = di['In'].get_shape().as_list()[1]
        W = tf.Variable(tf.random_normal([in_dim, m]))
        b = tf.Variable(tf.random_normal([m]))
        def fn(di):
            return {'Out' : tf.add(tf.matmul(di['In'], W), b)}
        return fn
    return htf.TFModule('Affine', {'m' : h_m}, cfn, ['In'], ['Out']).get_io()

def relu():
    return htf.TFModule('ReLU', {}, lambda di, dh: lambda di: {'Out' : tf.nn.relu(di['In'])},
        ['In'], ['Out']).get_io()

def dnn_block(h_m):
    return mo.siso_sequential([affine(h_m), relu()])

def get_ss1_fn(num_classes):
    def ss_fn():
        co.Scope.reset_default_scope()
        inputs, outputs = mo.siso_sequential([
            mo.siso_repeat(lambda: dnn_block(D([64, 128, 256, 512])), D([1, 2, 4])),
            affine(D([num_classes]))
        ])
        return inputs, outputs, {'lr' : D([1e-2, 1e-3, 1e-4])}
    return ss_fn
