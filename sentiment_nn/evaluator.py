import sys
sys.path.append('../')

import darch.core as co
import tensorflow as tf
from create_sentiment_featuresets import create_feature_sets_and_labels

def evaluate_fn(inputs, outputs, hs, data):
    train_x, train_y = data['train']
    val_x, val_y = data['val']

    x = tf.placeholder('float', [None, len(train_x[0])])
    y = tf.placeholder('float',)
    co.forward({inputs['In'] : x})
    prediction = outputs['Out'].val
    lr = hs['lr'].val if 'lr' in hs else 1e-3
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    hm_epochs = 100
    batch_size = 128
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                sess.run([optimizer], feed_dict = {x: batch_x, y: batch_y})
                i += batch_size

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        r = {'val_acc' : accuracy.eval({x: val_x, y:val_y}), 
             'train_acc' : accuracy.eval({x: train_x, y:train_y})}
        return r
