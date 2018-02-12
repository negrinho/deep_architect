
from __future__ import print_function

import research_toolbox.tb_utils as tb_ut
import research_toolbox.tb_training as tb_tr
import research_toolbox.tb_logging as tb_lg
import research_toolbox.tb_augmentation as tb_au
import research_toolbox.tb_io as tb_io
import research_toolbox.tb_random as tb_ra
import time

import darch.core as co
import darch.helpers.tensorflow as htf
import tensorflow as tf
import numpy as np
import pickle
import os
from pprint import pprint
import matplotlib.pyplot as plt

def batch_iterator(X_lst, batch_size):
    assert len(X_lst) > 0
    assert len(set([len(x) for x in X_lst])) == 1

    n = len(X_lst[0])
    num_batches = int(np.ceil(float(n) / batch_size))
    for idx_batch in xrange(num_batches):
        idx_start = idx_batch * batch_size
        idx_end = idx_start + batch_size

        Xbatch_lst = [X[idx_start:idx_end] for X in X_lst]
        yield Xbatch_lst

def batch_compute(sess, op, it, batch2feed_dict_fn):
    r_lst = []
    for x in it:
        d = batch2feed_dict_fn(x)
        r = sess.run(op, feed_dict=d)
        r_lst.append(r)
    return r_lst

def augment_data(X, is_train):
    X = tb_au.zero_pad_border(X, 10)
    if is_train:
        X = tb_au.random_flip_left_right(X, 0.5)
        X = tb_au.random_scale_rotate(X, 0, 360, 0.9, 1.1)
        X = tb_au.random_crop(X, 75, 75)
    else:
        X = tb_au.center_crop(X, 75, 75)
    # plt.imshow(X[0, :, :, 0])
    # plt.show()
    # plt.imshow(X[0, :, :, 1])    
    # plt.show()
    return X

def compute_accuracy(sess, eval_feed, x, y, num_correct, Xval, yval, batch_size):
    def batch2feed_dict_fn(b):
        eval_feed.update({x : b[0], y : b[1]})
        return eval_feed

    def it():
        for (Xbatch, ybatch) in batch_iterator([Xval, yval], batch_size):
            Xbatch = augment_data(Xbatch, False)
            yield (Xbatch, ybatch)

    rs = batch_compute(sess, num_correct, it(), batch2feed_dict_fn)
    return float(sum(rs)) / Xval.shape[0]

def compute_loss(sess, eval_feed, x, y, loss, Xdata, ydata, batch_size):
    def batch2feed_dict_fn(b):
        eval_feed.update({x : b[0], y : b[1]})
        return eval_feed

    def it():
        for (Xbatch, ybatch) in batch_iterator([Xdata, ydata], batch_size):
            Xbatch = augment_data(Xbatch, False)
            yield (Xbatch, ybatch)

    rs = batch_compute(sess, loss, it(), batch2feed_dict_fn)
    ndata = Xdata.shape[0]
    nbatches = len(rs)
    bs = [min(batch_size, ndata - i * batch_size) for i in xrange(nbatches)]
    rloss = (1.0 / ndata) * (np.array(rs, dtype='float') * np.array(bs, dtype='float')).sum()
    return rloss

def compute_probs(sess, eval_feed, x, probs, Xdata, batch_size):
    def batch2feed_dict_fn(b):
        eval_feed.update({x : b})
        return eval_feed

    def it():
        for (Xbatch,) in batch_iterator([Xdata], batch_size):
            Xbatch = augment_data(Xbatch, False)
            # print(Xbatch.shape)
            yield Xbatch

    rs = batch_compute(sess, probs, it(), batch2feed_dict_fn)
    return np.concatenate(rs)[:, 1]

def start_fn(d):
    (inputs, outputs, hs) = tb_ut.retrieve_values(d['darch'], 
        ['inputs', 'outputs', 'hs'], tuple_fmt=True)

    d['hs'] = {name : h.val for name, h in hs.iteritems()}
    dh = d['hs']
    dc = d['cfg']

    tf.reset_default_graph()
    tf.set_random_seed(0)

    x = tf.placeholder("float", [None] + list(d['cfg']['in_d']) )
    y = tf.placeholder("float", [None, d['cfg']['num_classes']])
    co.forward( {inputs['In'] : x} )
    logits = outputs['Out'].val 
    probs = tf.nn.softmax(logits, dim=1)
    train_feed, eval_feed = htf.get_feed_dicts(outputs.values()) 
        
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    if dh['weight_decay_coeff'] > 0.0:
        loss = loss + dh['weight_decay_coeff'] * tf.reduce_sum(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    x2y = tf.argmax(logits, 1)
    num_correct = tf.reduce_sum(tf.cast(tf.equal(x2y, tf.argmax(y, 1)), "float"))

    opt_type = dh['optimizer_type']
    lr = tf.placeholder(tf.float32)
    if opt_type == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    elif opt_type == 'sgd_mom':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=lr, momentum=d['cfg']['sgd_momentum'])
    else:
        raise ValueError("Unknown optimizer.")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = optimizer.minimize(loss)

    d['tf'] = {
        'x' : x, 'y' : y, 'logits' : logits, 'probs' : probs, 'lr' : lr, 
        'loss' : loss, 'x2y' : x2y, 'num_correct' : num_correct, 
        'optimizer' : optimizer,
        'train_feed' : train_feed, 'eval_feed' : eval_feed,
        'sess' : tf.Session(),
        'saver' : tf.train.Saver(),
    }

    # functions for the checkpoint.
    initial_acc = 0.0
    cond_fn = lambda old_acc, acc: acc > old_acc 
    def save_fn(acc):
        d['tf']['saver'].save(d['tf']['sess'], dc['model_path'])
        return acc
    def load_fn(acc):
        d['tf']['saver'].restore(d['tf']['sess'], dc['model_path'])
        return acc

    d['sch'] = {
        'lr' : tb_tr.CosineRateSchedule(dh['lr_start'], dh['lr_end'], dc['max_num_epochs']),
        'stop' : tb_tr.PatienceCounter(dh['stop_patience'], minimizing=False),
        'save' : tb_tr.Checkpoint(initial_acc, cond_fn, save_fn, load_fn),
        'timer' : tb_lg.TimeTracker()
    }
    
    d['log'] = {'train_loss' : [], 'val_loss' : [], 'train_acc' : [], 'val_acc' : [], 
        'lr' : [], 'time' : [], 'epoch' : []}

    # final stuff    
    d['tf']['sess'].run(tf.global_variables_initializer())

def train_fn(d):
    dc = d['cfg']
    timer = d['sch']['timer']
    max_minutes = dc['max_minutes_per_model']
    Xtrain, ytrain = d['data']['train']
    Xval, yval = d['data']['val']
    batch_size = dc['batch_size']

    (x, y, sess, optimizer, lr, loss, num_correct, train_feed, eval_feed
        ) = tb_ut.retrieve_values(d['tf'], 
            ['x', 'y', 'sess', 'optimizer', 'lr', 'loss', 
            'num_correct', 'train_feed', 'eval_feed'], tuple_fmt=True)

    ntrain = Xtrain.shape[0]
    nval = Xval.shape[0]

    lr_val = d['sch']['lr'].get_rate()
    train_loss = 0.0
    for (Xbatch, ybatch) in batch_iterator([Xtrain, ytrain], batch_size):
        Xbatch = augment_data(Xbatch, True)
        train_feed.update({x : Xbatch, y : ybatch, lr : lr_val})
        batch_loss, _ = sess.run([loss, optimizer], feed_dict=train_feed)
        train_loss += batch_loss * Xbatch.shape[0]
    
        if timer.time_since_start('m') > max_minutes:
            break

    results_d = {
        'epoch' : len(d['log']['lr']),
        'train_loss' : train_loss / ntrain,
        'val_loss' : compute_loss(sess, eval_feed, x, y, loss, Xval, yval, batch_size),        
        'train_acc' : compute_accuracy(sess, eval_feed, x, y, num_correct, Xtrain, ytrain, batch_size),
        'val_acc' : compute_accuracy(sess, eval_feed, x, y, num_correct, Xval, yval, batch_size),
        'lr' : lr_val,
        'time' : d['sch']['timer'].time_since_start('s')
    }
    for k, v in results_d.iteritems():
        d['log'][k].append(v)
    pprint(results_d)

    v = results_d['val_acc']
    d['sch']['lr'].update(v)
    d['sch']['stop'].update(v)
    d['sch']['save'].update(v)

    tb_ra.shuffle_tied([Xtrain, ytrain])

def is_over_fn(d):
    ds = d['sch']
    dc = d['cfg']

    return ds['stop'].has_stopped() or (
        ds['timer'].time_since_start('m') > dc['max_minutes_per_model']) or (
        len(d['log']['lr']) >= dc['max_num_epochs'])
    
def end_fn(d):
    d['sch']['save'].get()
    Xtest = d['data']['test']

    dd = d['data']
    dt = d['tf']
    dc = d['cfg']

    ps = compute_probs(dt['sess'], dt['eval_feed'], dt['x'], dt['probs'], dd['test'], dc['batch_size'])

    lines = ['id,is_iceberg']
    for (k, p) in zip(dd['test_ids'], ps):
        lines.append('%s,%f' % (k, p))
    tb_io.write_textfile(dc['preds_path'], lines, with_newline=True)

    dt['sess'].close()
    # TODO: add other stuff and potentially change things.

def get_eval_fn():
    return tb_tr.get_eval_fn(start_fn, train_fn, is_over_fn, end_fn)