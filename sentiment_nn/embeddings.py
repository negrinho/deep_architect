# from __future__ import print_function

import sys
sys.path.append('../')

from six.moves import xrange, zip
import numpy as np
import pickle
import matplotlib.pyplot as plt
import darch.surrogates as su

import torch
import torch.nn as nn
import torch.nn.functional as fnc
from torch.autograd import Variable
import torch.optim as opt
from pprint import pprint
import itertools 
import numpy as np
import random

def get_ch2idx():
    ch_lst = [chr(i) for i in range(ord('A'), ord('Z'))] + [
        chr(i) for i in range(ord('a'), ord('z') + 1)] + [
        chr(i) for i in range(ord('0'), ord('9') + 1)] + [
            '.', ':', '-', '_', '<', '>', '/', '=', '*', ' ', '|']
    return {ch : idx for (idx, ch) in enumerate(ch_lst)}

# TODO: come up with better names.
def process_to_same_length(ch2idx, xs):
    maxlen = max([len(x) for x in xs]) + 2

    vec_lst = []
    for x in xs:
        # initial character
        vec = [ch2idx['*']]
        # actual string
        for ch in x:
            vec.append(ch2idx[ch])
        # rightmost padding
        while len(vec) < maxlen:
            vec.append(ch2idx['*'])
        vec_lst.append(vec)
    return vec_lst

def compute_pred(surr_model, feats_lst):
    return [surr_model.eval(f) for f in feats_lst]
    
def compute_error(surr_model, val_lst, feats_lst):
    av = 0.0
    for v, feats in zip(val_lst, feats_lst):
        v_hat = surr_model.eval(feats)
        av += (v - v_hat) * (v - v_hat)
    return av / len(val_lst)

def concatenate_features(feats):
    xs = []
    for f in feats:
        xs.extend(f)
    return xs

def plot_true_vs_pred(true_val_lst, hat_val_lst):
    plt.scatter(true_val_lst, hat_val_lst)
    plt.plot([0, 1], [0, 1])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()

class CLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        nn.Module.__init__(self)

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.ch2idx = get_ch2idx()
        self.ch_embs = nn.Embedding(len(self.ch2idx), embedding_size)

        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, char_repr):
        out = self.ch_embs(char_repr)
        (h0, c0) = (Variable(torch.zeros( 1, out.size(1), out.size(2) )), 
            Variable(torch.zeros( 1, out.size(1), out.size(2) )))            

        _, (_, out) = self.lstm(out, (h0, c0))
        out = out.mean(1)
        out = self.fc(out)
        return out

class CLSTMSurrogate(su.SurrogateModel):
    def __init__(self, embedding_size, hidden_size, learning_rate, num_epochs):
        self.clstm = CLSTM(embedding_size, hidden_size)
        self.mse = nn.MSELoss()
        self.optimizer = opt.Adam(self.clstm.parameters(), learning_rate)
        self.ch2idx = get_ch2idx()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.feats_lst = []
        self.val_lst = []


    def eval(self, feats):
        vec_lst = process_to_same_length(self.ch2idx, concatenate_features(feats))
        vec = Variable( torch.LongTensor(vec_lst).t() )
        return self.clstm.forward(vec).data[0]

    def update(self, val, feats):
        self.val_lst.append(val)
        self.feats_lst.append(feats)

    def _refit(self):
        val_frac = 0.5
        num_total = len(self.val_lst)
        num_train = int((1.0 - val_frac) * num_total)

        train_val_lst = self.val_lst[:num_train]
        train_feats_lst = self.feats_lst[:num_train]
        val_val_lst = self.val_lst[num_train:]
        val_feats_lst = self.feats_lst[num_train:]
        idxs = range(len(train_val_lst))

        print "Train: %f, Val: %f" % (
            compute_error(self, train_val_lst, train_feats_lst), 
            compute_error(self, val_val_lst, val_feats_lst))

        for _ in xrange(self.num_epochs):
            random.shuffle(idxs)
            for i in idxs:
                val, feats = train_val_lst[i], train_feats_lst[i]
                feats = concatenate_features(feats)
                vec_lst = process_to_same_length(self.ch2idx, feats)
                vec = Variable(torch.LongTensor(vec_lst).t())
                y = Variable(torch.FloatTensor([[val]]))
                out = self.clstm(vec)
                loss = self.mse(out, y)
                # print loss.data[0]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print "Train: %f, Val: %f" % (
                compute_error(self, train_val_lst, train_feats_lst), 
                compute_error(self, val_val_lst, val_feats_lst))

if __name__ == '__main__':
    data = pickle.load(open('perf-pred/evals_0.2.pkl', 'rb'))
    val_lst = data['v_lst']
    feats_lst = data['feats_lst']

    val_frac = 0.5
    num_total = len(val_lst)
    num_train = int((1.0 - val_frac) * num_total)

    train_val_lst = val_lst[:num_train]
    train_feats_lst = feats_lst[:num_train]
    val_val_lst = val_lst[num_train:]
    val_feats_lst = feats_lst[num_train:]

    hash_size = 1024
    refit_interval = 1e6
    for weight_decay_coeff in [10.0]:#[1e-4, 1e-5, 1e-6]:
        surr_model = su.HashingSurrogate(hash_size, refit_interval, 
            weight_decay_coeff=weight_decay_coeff)
        for val, feats in zip(train_val_lst, train_feats_lst):
            surr_model.update(val, feats)
        surr_model._refit()
        print "Train MSE: %f, Val MSE: %f" % (
            compute_error(surr_model, train_val_lst, train_feats_lst), 
            compute_error(surr_model, val_val_lst, val_feats_lst))
        plot_true_vs_pred(val_val_lst, compute_pred(surr_model, val_feats_lst))        
        

### NOTE: the linear type models are easier to fit

    embedding_size = 128
    hidden_size = 128
    learning_rate = 1e-4
    num_epochs = 100
    surr_model = CLSTMSurrogate(embedding_size, hidden_size, learning_rate, num_epochs)
    for val, feats in zip(val_lst, feats_lst):
        surr_model.update(val, feats)    
    surr_model._refit()

    

# TODO: maybe do bidirectional stuff.
# NOTE: do a very simple regression problem. consider a ranking problem
# NOTE: this depends on what the embeddings are going to be used for. 
# sample a few features and a few numbers. 
# check both the existing surrogates and surrogates that I may consider having.
# TODO: do architecture search for the surrogate itself. example, better models 
# may work best.
# TODO: check that a full LSTM would also work.
# TODO: generate the embeddings that I can give to Yiming.
# TODO: add a simple neural network with two layers.
# TODO: add code to the test the performance of the surrogates in performance prediction.
# TODO: profile the code. what takes more time: feature extraction or computation.
# TODO: consider a few variations of the model, like adjusting the exploration
# bonus automatically (for optimization in MCTS case).
# TODO: change the name vec_lst
# TODO: more careful separation of the features.