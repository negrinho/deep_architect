import sys
sys.path.append('../')

import darch.core as co
import darch.hyperparameters as hp
import darch.helpers.tensorflow as htf
import darch.modules as mo
import darch.utils as ut
import darch.searchers as se
import darch.visualization as vi

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from create_sentiment_featuresets import create_feature_sets_and_labels
import numpy as np

D = hp.Discrete

def evaluate_fn(inputs, outputs, hs):
    x = tf.placeholder('float', [None, len(train_x[0])])
    y = tf.placeholder('float', )

    co.forward( {inputs['In'] : x} )
    prediction = outputs['Out'].val

    if 'lr' in hs:
        lr = hs['lr'].val
    else:
        lr = 1e-3

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    # cycles of feed forward + backprop
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                # ^^^ how does this work
                epoch_loss += c
                i += batch_size
            # print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        r = {'val_acc' : accuracy.eval({x: val_x, y:val_y}),
             'test_acc' : accuracy.eval({x:test_x, y:test_y})}
        return r

def Affine(h_m):
    def cfn(In, m):
        in_dim = In.get_shape().as_list()[1]
        W = tf.Variable(tf.random_normal([in_dim, m]))
        b = tf.Variable(tf.random_normal([m]))
        def fn(In):
            return {'Out' : tf.add(tf.matmul(In, W), b)}
        return fn
    return htf.TFModule('Affine', {'m' : h_m}, cfn, ['In'], ['Out'])

def ReLU():
    return htf.TFModule('ReLU', {}, lambda: lambda In: {'Out' : tf.nn.relu(In)},
        ['In'], ['Out'])

def search_space_model_fn():
    co.Scope.reset_default_scope()
    ms = [
        Affine( D([ 500 ]) ),
        ReLU(),
        Affine( D([ 500 ]) ),
        ReLU(),
        Affine( D([ 500 ]) ),
        ReLU(),
        Affine( D([ n_classes ]) )
    ]
    for (m, m_next) in zip(ms[0:-1], ms[1:]):
        m.outputs['Out'].connect(m_next.inputs['In'])

    return ms[0].inputs, ms[-1].outputs, {}

def ss0_fn():
    co.Scope.reset_default_scope()
    ms = [
        Affine( D([ 64, 128, 256, 512 ]) ),
        ReLU(),
        Affine( D([ 64, 128, 256, 512 ]) ),
        ReLU(),
        Affine( D([ 64, 128, 256, 512 ]) ),
        ReLU(),
        Affine( D([ n_classes ]) )
    ]
    for (m, m_next) in zip(ms[0:-1], ms[1:]):
        m.outputs['Out'].connect(m_next.inputs['In'])

    return ms[0].inputs, ms[-1].outputs, {'lr' : D([ 1e-2, 1e-3, 1e-4])}

def io_DNNBlock(h_m):
    x = Affine(h_m)
    y = ReLU()
    x.outputs['Out'].connect(y.inputs['In'])
    return x.inputs, y.outputs

def ss1_fn():
    co.Scope.reset_default_scope()

    ms = [
        mo.SISORepeat(lambda: io_DNNBlock( D([ 64, 128, 256, 512 ]) ), D([ 1, 2, 4 ])),
        Affine( D([ n_classes ]) )
    ]
    ut.connect_sequentially(ms)
    return ms[0].inputs, ms[-1].outputs, {'lr' : D([ 1e-2, 1e-3, 1e-4])}

if __name__ == '__main__':
    n_classes = 2
    batch_size = 100 # batches of 100 features at a time

    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    train_size = int(len(train_x)*.8)

    train_x, val_x = np.array(train_x[:train_size]), np.array(train_x[train_size:])
    train_y, val_y = np.array(train_y[:train_size]), np.array(train_y[train_size:])

    searcher = se.MCTSearcher(ss1_fn, 0.1)
    for _ in xrange(128):
        (inputs, outputs, hs, vs, cfg_d) = searcher.sample()
        r = evaluate_fn(inputs, outputs, hs)
        print vs, r, cfg_d
        searcher.update(r['val_acc'], cfg_d)
        # vi.draw_graph(outputs.values(), True)
