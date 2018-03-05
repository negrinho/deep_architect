
# this file will be used to explore the creation of embeddings with the 
# desired properties. what is the meaning of embedding a function into 
# \mathbb R ^2

#%%

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
import matplotlib.pyplot as plt

def compute_mse(y, y_other):
    err = y - y_other
    mse = (err * err).sum() / err.shape[0]
    return mse

def f0(x):
    return x.sum(axis=1)

def f1(x):
    return np.abs(x * np.sin(x) + 0.1 * x).sum(axis=1)

#%% Simple model with performance prediction.

class Model(nn.Module):
    def __init__(self, d_in, d_hidden, d_emb):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_emb)
        # self.fc3 = nn.Linear(d_emb, 1)
    
    def forward(self, x):
        out = self.fc2(F.relu(self.fc1(x)))
        # out = self.fc3(F.relu(out))
        return out

d_in = 10
d_hidden = 128
d_emb = 2
learning_rate = 1e-3
model = Model(d_in, d_hidden, d_emb)
optimizer = opt.Adam(model.parameters(), learning_rate)
mse = nn.MSELoss()

n = 1024
num_epochs = 100
num_train = int(0.9 * n)
X = np.random.rand(n, d_in)
y = f1(X)
X_th = torch.FloatTensor(X)
y_th = torch.FloatTensor(y)
for _ in xrange(num_epochs):
    X_var = Variable(X_th[:num_train])
    y_var = Variable(y_th[:num_train])
    y_pred_var = model(X_var).sum(1)
    loss = mse(y_pred_var, y_var)
    optimizer.zero_grad()
    loss.backward()
    print loss.data[0]
    optimizer.step()

embs = model(Variable(X_th[num_train:], volatile=True)).data.numpy()
vals = y[num_train:]
print 'MSE', compute_mse(embs.sum(axis=1), vals)

plt.scatter(embs[:, 0], embs[:, 1], c=vals, cmap='viridis')
plt.colorbar()
plt.show()

#%%%

### compare model.
d_in = 100
d_hidden = 128
d_emb = 32
learning_rate = 1e-3
emb_model = Model(d_in, d_hidden, d_emb)
comp_model = Model(2 * d_emb, d_hidden, 1)
param_lst = list(emb_model.parameters()) + list(comp_model.parameters())
optimizer = opt.Adam(param_lst, learning_rate)
bce = nn.BCEWithLogitsLoss()

n = 1024
num_epochs = 100
num_train = int(0.9 * n)
X = np.random.rand(n, d_in)
y = f1(X)
X_th = torch.FloatTensor(X)
y_th = torch.FloatTensor(y)
for _ in xrange(num_epochs):
    idxs = np.arange(num_train)
    np.random.shuffle(idxs)

    X_var = Variable(X_th[:num_train])
    X_other_var = Variable(X_th[:num_train][idxs])
    y_class = Variable(
        (y_th[:num_train] < y_th[:num_train][idxs]).float())
    embs = emb_model(X_var)
    embs_other = emb_model(X_other_var)
    out = torch.cat([embs, embs_other], dim=1)
    out = comp_model(out).squeeze()
    loss = bce.forward(out, y_class)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print loss.data[0]

embs = emb_model(Variable(X_th[num_train:], volatile=True)).data.numpy()
vals = y[num_train:]
plt.scatter(embs[:, 0], embs[:, 1], c=vals, cmap='viridis')
plt.colorbar()
plt.show()



#%%
# print y[num_train:]

# NOTE: this is interesting, but I don't think that it is doing anything 
# particular interesting, as the model is simply printing things in a line.


# TODO: add some more visualization tools.

#*** Embedding criterions (measuring whether two inputs are similar or dissimilar):
# HingeEmbeddingCriterion: takes a distance as input;
# L1HingeEmbeddingCriterion: L1 distance between two inputs;
# CosineEmbeddingCriterion: cosine distance between two inputs;
# DistanceRatioCriterion: Probabilistic criterion for training siamese model with triplets.

### TODO: can also focus on visualizations here.
# TODO: possible to consider higher level and then some form of dimensionality 
# reduction.
# TODO: treat it as a classification problem where we are looking at 
# quantiles.
# or perhaps, just try to classify it in different quantiles.