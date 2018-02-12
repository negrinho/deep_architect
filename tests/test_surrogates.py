from __future__ import print_function
from six.moves import xrange, zip
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import darch.searchers as se
import tests.search_space as ss
import tests.dataset as ds
import tests.evaluator as ev
import reduced_toolbox.tb_logging as lg

def generate_data(num_evals, max_minutes_per_model):
    # loading the dataset and evaluator
    (train_dataset, val_dataset, test_dataset, in_d, num_classes) = ds.load_data(
        'mnist')

    evaluator = ev.ClassifierEvaluator(train_dataset, val_dataset, in_d, num_classes, 
        'temp/', 'mnist', 
        output_to_terminal=True, test_dataset=test_dataset,
        max_minutes_per_model=max_minutes_per_model, batch_size_init=128)

    rand_se = se.RandomSearcher(ss.search_space_fn)

    d = { 'feats_lst' : [], 'v_lst' : [], 'v_hist_lst' : [] }
    for _ in xrange(num_evals):
        (inputs, outputs, hs, v_hist, cfg_d) = rand_se.sample()
        v = evaluator.eval(inputs, outputs, hs)
        rand_se.update(v, cfg_d)

        d['feats_lst'].append( se.extract_features(inputs, outputs, hs) )
        d['v_lst'].append( v )
        d['v_hist_lst'].append( v_hist )
        
    with open('data/perf-pred/evals_%0.1f.pkl' % max_minutes_per_model, 'wb') as f:
        pickle.dump(d, f)

def eval_fit():
    d = pickle.load(open('data/perf-pred/evals_0.1.pkl', 'rb'))
    v_lst, feats_lst = d['v_lst'], d['feats_lst']
    
    n = int(len(v_lst) / 2)
    train_v_lst, train_feats_lst = v_lst[:n], feats_lst[:n]
    val_v_lst, val_feats_lst = v_lst[n:], feats_lst[n:]

    weight_coeff_lst = [1e-1, 1e-3, 1e-5]
    for wc in weight_coeff_lst:
        h_sur = se.HashingSurrogate(16 * 1024, 100000, wc)

        for v, feats in zip(train_v_lst, train_feats_lst):
            h_sur.update(v, feats)
        h_sur._refit()

        v_hat_lst = []
        for v, feats in zip(val_v_lst, val_feats_lst):
            v_hat = h_sur.eval(feats), 
            v_hat_lst.append(v_hat)
        
        plt.plot([0, 1], [0, 1])
        plt.scatter(val_v_lst, v_hat_lst)
        plt.show()

def eval_maximize():
    d = pickle.load(open('data/perf-pred/evals_0.1.pkl', 'rb'))
    v_lst, feats_lst = d['v_lst'], d['feats_lst']
    
    h_sur = se.HashingSurrogate(4 * 1024, 100000, 1e-5)

    for v, feats in zip(v_lst, feats_lst):
        h_sur.update(v, feats)
    h_sur._refit()

    timer = lg.TimeTracker()

    num_evals = 1024
    rand_v_lst = []
    for _ in xrange(num_evals):
        (inputs, outputs, hs) = ss.search_space_fn()
        se.random_specify(outputs.values(), hs.values())
        feats = se.extract_features(inputs, outputs, hs)
        v = h_sur.eval(feats)
        rand_v_lst.append(v)
    rand_time = timer.time_since_last()
    
    smbo_mcts_v_lst = []
    sm_searcher = se.SMBOSearcherWithMCTSOptimizer(ss.search_space_fn, h_sur, 1, 0.1, 1000000)
    for _ in xrange(num_evals):
        (inputs, outputs, hs, _, _) = sm_searcher.sample()
        feats = se.extract_features(inputs, outputs, hs)
        v = h_sur.eval(feats)
        smbo_mcts_v_lst.append(v)
    smbo_mcts_time = timer.time_since_last()
    
    print("rand: %f, mcts_smbo: %f" % (rand_time, smbo_mcts_time))
    plt.plot( np.maximum.accumulate(rand_v_lst), label='rand')
    plt.plot( np.maximum.accumulate(smbo_mcts_v_lst), label='smbo_mcts')
    plt.legend()
    plt.show()

def test_clstm_surrogate():
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    import torch.optim as opt

    ch_lst = [chr(i) for i in range(ord('A'), ord('Z'))] + [
        chr(i) for i in range(ord('a'), ord('z') + 1)] + [
        chr(i) for i in range(ord('0'), ord('9') + 1)] + [
            '.', ':', '-', '_', '<', '>', '/', '=', '*', ' ', '|']
    ch2idx = {ch : idx for (idx, ch) in enumerate(ch_lst)}

    hidden_size = 128
    embedding_size = 128
    
    # requirements: should be fast to evaluate; should be able to make use of 
    # patterns that lead to good performance.
    # TODO: try different types of modules.
    class Model(nn.Module):
        def __init__(self, embedding_size, hidden_size):
            nn.Module.__init__(self)

            self.ch_embs = nn.Embedding(len(ch_lst), embedding_size)
            self.lstm = torch.nn.LSTM(embedding_size, hidden_size, 1)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, vec):
            out = self.ch_embs(vec)
            (h0, c0) = (Variable(torch.zeros( 1, out.size(1), out.size(2) )), 
                Variable(torch.zeros( 1, out.size(1), out.size(2) )))            

            _, (_, out) = self.lstm( out, (h0, c0))
            out = out.mean(1)
            out = self.fc(out)
            return out

    mdl = Model(embedding_size, hidden_size)
    mse = nn.MSELoss()
    optimizer = opt.Adam( mdl.parameters(), 1e-3 )

    def process(feats):
        xs = feats[0] + feats[1] + feats[2] + feats[3]
        maxlen = max([len(x) for x in xs]) + 2

        vec_lst = []
        for x in xs:
            vec = [ ch2idx['*'] ]
            for ch in x:
                vec.append( ch2idx[ch] )
            
            while len(vec) < maxlen:
                vec.append( ch2idx['*'] )
            
            vec_lst.append(vec)
        return vec_lst

    d = pickle.load(open('evals_0.1.pkl', 'rb'))

    n = 1024
    v_lst = d['v_lst'][:n]
    feats_lst = d['feats_lst'][:n]
    val_v_lst = d['v_lst'][n:2*n]
    val_feats_lst = d['feats_lst'][n:2*n]
    idxs = range(len(v_lst))

    def error_fn(h_sur, v_lst, feats_lst):
        av = 0.0
        for v, feats in zip(v_lst, feats_lst):
            h_sur.update(v, feats)
            v_hat = h_sur.eval(feats)
            av += (v - v_hat) * (v - v_hat)
        return av / len(v_lst)

    def error_other_fn(mdl, v_lst, feats_lst):
        av = 0.0
        for v, feats in zip(v_lst, feats_lst):
            vec_lst = process(feats)
            vec = Variable( torch.LongTensor(vec_lst).t() )
            v_hat = mdl(vec).data[0, 0]
            av += (v - v_hat) * (v - v_hat)
        return av / len(v_lst)        

    av_bef = 0.0
    h_sur = se.HashingSurrogate(1024, 100000)
    h_sur_other = se.HashingSurrogate(1024 * 16, 100000)
    
    print(error_fn(h_sur, v_lst, feats_lst), error_fn(h_sur, val_v_lst, val_feats_lst))
    print(error_fn(h_sur_other, v_lst, feats_lst), error_fn(h_sur_other, val_v_lst, val_feats_lst))

    for v, feats in zip(v_lst, feats_lst):
        h_sur.update(v, feats)
        h_sur_other.update(v, feats)
    
    h_sur._refit()
    h_sur_other._refit()

    print(error_fn(h_sur, v_lst, feats_lst), error_fn(h_sur, val_v_lst, val_feats_lst))
    print(error_fn(h_sur_other, v_lst, feats_lst), error_fn(h_sur_other, val_v_lst, val_feats_lst))
    
    for _ in xrange(1000):
        random.shuffle(idxs)
        for i in idxs:
            optimizer.zero_grad()

            v, feats = v_lst[i], feats_lst[i]
            vec_lst = process(feats)
        
            vec = Variable( torch.LongTensor(vec_lst).t() )
            y = Variable( torch.FloatTensor([[v]]))
            out = mdl(vec)

            loss = mse(out, y)
            loss.backward()
            optimizer.step()
        
        print(error_other_fn(mdl, v_lst, feats_lst), error_other_fn(mdl, val_v_lst, val_feats_lst))

if __name__ == '__main__':
    # generate_data(128, 0.1)
    # generate_data(2048, 0.2)
    # generate_data(2048, 0.4)
    # main()
    # eval_fit()
    eval_maximize()
