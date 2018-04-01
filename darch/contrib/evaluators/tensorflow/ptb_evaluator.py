import sys

import core as co
import tensorflow as tf
import math

def evaluate_fn(inputs, outputs, hs, data, vocab_size, batch_size, num_steps):
    tf.reset_default_graph()
    train_x, train_y = data['train']
    val_x, val_y = data['val']
    
    batch_size = 128
    num_steps = 35
    x = tf.placeholder('int32', [None, train_x.shape[2]])
    y = tf.placeholder('int32', [None, train_y.shape[2]])
    y_one_hot = tf.one_hot(y, vocab_size)
    co.forward({inputs['In'] : x})
    prediction = outputs['Out'].val
    lr = hs['lr'].val if 'lr' in hs else 1e-3
    cost = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=prediction, targets=y, weights=tf.ones([batch_size, num_steps])))
    correct = tf.equal(tf.argmax(prediction, 2, output_type=tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    hm_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            loss = 0.
            for i in range(0, len(train_x)):
                batch_x = train_x[i]
                batch_y = train_y[i]
                _, cost_val = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                loss += cost_val
            print 'Epoch %d: %f' % (epoch, math.exp(loss/len(train_x)))

        
	val_acc = 0.
	val_loss = 0.
	for i in range(0, len(val_x)):
	    batch_x = val_x[i]
	    batch_y = val_y[i]
            _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict = {x: batch_x, y: batch_y})
            val_loss += cost_val
	    val_acc += acc_val
        r = {'val_perp' : math.exp(val_loss/len(val_x)),
	     'val_acc' : val_acc / len(val_x)
             }
        return r
