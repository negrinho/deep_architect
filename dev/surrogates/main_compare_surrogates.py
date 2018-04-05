import darch.surrogates as su
import darch.search_logging as sl
import darch.visualization as vi
from rnn_surrogates import RankingRNNSurrogate, RNNSurrogate, CharLSTMSurrogate
from hashing_surrogates import SimplerHashingSurrogate

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random

import itertools
from collections import OrderedDict

name_to_cfg = {
    'debug' : {
        'embedding_size' : 16,
        'hidden_size' : 16,
        'experiment_type' : 'train_unbatched_ranking_charlstm',
        'num_val' : None,
        'num_train' : None
    },
}
cfg_name = 'debug'
cfg = name_to_cfg[cfg_name]

def compute_rmse(surr_model, feats_lst, acc_lst):
    err = 0.0
    for feats, acc in itertools.izip(feats_lst, acc_lst):
        pred_acc = surr_model.eval(feats)
        err += (pred_acc - acc) ** 2
    return np.sqrt(err / len(feats_lst))

def compute_ranking_error(surr_model, feats_lst, acc_lst, num_pairs):
    err = 0.0
    n = len(feats_lst)
    for _ in xrange(num_pairs):
        i = np.random.randint(n)
        i_other = np.random.randint(n)

        score = surr_model.eval(feats_lst[i])
        score_other = surr_model.eval(feats_lst[i_other])
        # print score, score_other

        acc = acc_lst[i]
        acc_other = acc_lst[i_other]
        if ((score > score_other) and (acc < acc_other)) or (
            (score < score_other) and (acc > acc_other)):
            err += 1.0
    return err / num_pairs

def train(surr_model, feats_lst, acc_lst):
    for feats, acc in itertools.izip(feats_lst, acc_lst):
        surr_model.update(acc, feats)

def predict(surr_model, feats_lst):
    return [surr_model.eval(feats) for feats in feats_lst]

def plot_scatter(acc_lst, pred_acc_lst):
    plt.scatter(acc_lst, pred_acc_lst)
    plt.plot([0, 1], [0, 1])
    plt.xlabel('true accuracy')
    plt.ylabel('predicted accuracy')
    plt.show()

def save_plot(filename):
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')

def get_data(randomize=False, num_train=None, num_val=None):
    acc_key = 'validation_accuracy'
    # log_lst = sl.read_search_folder('./logs/benchmark_surrogates.small.mnist')
    # log_lst = sl.read_search_folder('./logs/cifar10_medium/run-0')
    log_lst = sl.read_search_folder('./logs/train'); acc_key = 'val_acc'
    if randomize:
        random.shuffle(log_lst)
    dataset_size = len(log_lst)

    feats_lst = [x['features'] for x in log_lst]
    acc_lst = [x['results'][acc_key] for x in log_lst]

    validation_size = int(dataset_size / 2)
    train_feats_lst, val_feats_lst = (feats_lst[validation_size:], feats_lst[:validation_size])
    train_acc_lst, val_acc_lst = (acc_lst[validation_size:], acc_lst[:validation_size])

    if num_train is not None:
        train_feats_lst = train_feats_lst[:num_train]
        train_acc_lst = train_acc_lst[:num_train]
    if num_val is not None:
        val_feats_lst = val_feats_lst[:num_val]
        val_acc_lst = val_acc_lst[:num_val]
    return (train_feats_lst, train_acc_lst, val_feats_lst, val_acc_lst)

def train_batched_charlstm():
    train_feats_lst, train_acc_lst, val_feats_lst, val_acc_lst = get_data(
        num_train=cfg['num_train'], num_val=cfg['num_val'])
    num_refits = 100
    clstm_sur = CharLSTMSurrogate(refit_interval=10000)

    for feats, acc in itertools.izip(train_feats_lst, train_acc_lst):
        clstm_sur.update(acc, feats)

    for i in xrange(num_refits):
        train_rmse = compute_rmse(clstm_sur, train_feats_lst, train_acc_lst)
        val_rmse = compute_rmse(clstm_sur, val_feats_lst, val_acc_lst)
        print "refit %d <> train: %f, val: %f" % (i, train_rmse, val_rmse)
        clstm_sur._refit()

def train_unbatched_charlstm():
    train_feats_lst, train_acc_lst, val_feats_lst, val_acc_lst = get_data(
        num_train=cfg['num_train'], num_val=cfg['num_val'])
    num_refits = 100
    clstm_sur = RankingRNNSurrogate(cfg['embedding_size'], cfg['hidden_size'])

    dummy_sur = su.DummySurrogate()
    for feats, acc in itertools.izip(train_feats_lst, train_acc_lst):
        clstm_sur.update(acc, feats)
        dummy_sur.update(acc, feats)

    train_rmse = compute_rmse(dummy_sur, train_feats_lst, train_acc_lst)
    val_rmse = compute_rmse(dummy_sur, val_feats_lst, val_acc_lst)
    print "dummy <> train: %f, val: %f" % (train_rmse, val_rmse)
    for i in xrange(num_refits):
        train_rmse = compute_rmse(clstm_sur, train_feats_lst, train_acc_lst)
        val_rmse = compute_rmse(clstm_sur, val_feats_lst, val_acc_lst)
        print "refit %d <> train: %f, val: %f" % (i, train_rmse, val_rmse)
        clstm_sur._refit()

def train_unbatched_ranking_charlstm():
    train_feats_lst, train_acc_lst, val_feats_lst, val_acc_lst = get_data(
        num_train=cfg['num_train'], num_val=cfg['num_val'])
    num_refits = 100
    num_pairs = 100
    clstm_sur = RankingRNNSurrogate(cfg['embedding_size'], cfg['hidden_size'])

    for feats, acc in itertools.izip(train_feats_lst, train_acc_lst):
        clstm_sur.update(acc, feats)

    for i in xrange(num_refits):
        train_err = compute_ranking_error(clstm_sur, train_feats_lst, train_acc_lst, num_pairs)
        val_err = compute_ranking_error(clstm_sur, val_feats_lst, val_acc_lst, num_pairs)
        print "refit %d <> train: %f, val: %f" % (i, train_err, val_err)
        clstm_sur._refit()

def compare_surrogates_mse():
    train_feats_lst, train_acc_lst, val_feats_lst, val_acc_lst = get_data()

    plotter = vi.LinePlot()
    name_to_get_surr_model_fn = OrderedDict([
        ('dummy', lambda: su.DummySurrogate()),
        ('hash_simpler_b1024', lambda: SimplerHashingSurrogate(1024, refit_interval=1, weight_decay_coeff=1e-5)),
        ('hash_b1024', lambda: su.HashingSurrogate(1024, refit_interval=1, weight_decay_coeff=1e-5)),
        # ('hash_b2048', lambda: su.HashingSurrogate(2048, refit_interval=1, weight_decay_coeff=1e-5)),
        # ('hash_b4096', lambda: su.HashingSurrogate(4096, refit_interval=1, weight_decay_coeff=1e-5)),
    ])
    # for t in itertools.product([0, 1], [0, 1], [0, 1], [0, 1]):
    #     d = dict(zip([
    #         'use_module_feats', 'use_connection_feats',
    #         'use_module_hyperp_feats', 'use_other_hyperp_feats'], t))
    #     if any(t):
    #         name_to_get_surr_model_fn['hash_b1024_%s' % "".join(map(str, t))
    #             ] = lambda: su.HashingSurrogate(1024, refit_interval=1, weight_decay_coeff=1e-5, **d)

    for name, fn in name_to_get_surr_model_fn.iteritems():
        surr_model = fn()
        n_prev = 0
        xs = []
        ys = []
        for n in [2 ** i for i in xrange(1, 20)]:
            train(surr_model, train_feats_lst[n_prev:n], train_acc_lst[n_prev:n])
            n = len(train_feats_lst[:n])
            n_prev = n
            train_rmse = compute_rmse(surr_model, train_feats_lst[:n], train_acc_lst[:n])
            val_rmse = compute_rmse(surr_model, val_feats_lst, val_acc_lst)

            xs.append(n)
            ys.append(val_rmse)
            print "%s, %d <> train: %f, val: %f" % (name, n, train_rmse, val_rmse)
            if n >= len(train_feats_lst):
                break
        # plot_scatter(val_acc_lst, predict(surr_model, val_feats_lst))
        plotter.add_line(xs, ys, label=name)
    plotter.plot(fpath='./surrogate_comparison_mse.png')

def compare_surrogates_ranking():
    train_feats_lst, train_acc_lst, val_feats_lst, val_acc_lst = get_data()
    num_pairs = 1000

    plotter = vi.LinePlot()
    name_to_get_surr_model_fn = OrderedDict()
    for embedding_size in [16, 32, 64, 128, 256]:
        name_to_get_surr_model_fn['clstm_%d' % embedding_size] = lambda: RankingRNNSurrogate(
            embedding_size, embedding_size)

    for name, fn in name_to_get_surr_model_fn.iteritems():
        surr_model = fn()
        n_prev = 0
        xs = []
        ys = []
        for n in [2 ** i for i in xrange(1, 20)]:
            train(surr_model, train_feats_lst[n_prev:n], train_acc_lst[n_prev:n])
            n = len(train_feats_lst[:n])
            n_prev = n
            train_err = compute_ranking_error(surr_model, train_feats_lst[:n], train_acc_lst[:n], num_pairs)
            val_err = compute_ranking_error(surr_model, val_feats_lst, val_acc_lst, num_pairs)

            xs.append(n)
            ys.append(val_err)
            print "%s, %d <> train: %f, val: %f" % (name, n, train_err, val_err)
            if n >= len(train_feats_lst):
                break
        # plot_scatter(val_acc_lst, predict(surr_model, val_feats_lst))
        plotter.add_line(xs, ys, label=name)
    plotter.plot(fpath='./surrogate_comparison_ranking.png')

if __name__ == '__main__':
    name_to_experiment = {
        'train_unbatched_charlstm' : train_unbatched_charlstm,
        'train_unbatched_ranking_charlstm' : train_unbatched_ranking_charlstm,
        'train_batched_charlstm' : train_batched_charlstm,
        'compare_surragates_mse' : compare_surrogates_mse,

    }
    name_to_experiment[cfg['experiment_type']]()