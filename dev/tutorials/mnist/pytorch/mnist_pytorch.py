# Search Space
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
from deep_architect.helpers.pytorch import siso_pytorch_module

D = hp.Discrete  # Discrete Hyperparameter


def flatten():

    def compile_fn(di, dh):
        shape = di['In'].size()
        n = np.product(shape[1:])

        def fn(di):
            return {'Out': (di['In']).view(-1, n)}

        return fn, []

    return siso_pytorch_module('Flatten', compile_fn, {})


def dense(h_units):

    def compile_fn(di, dh):
        (_, in_dim) = di['In'].size()
        Dense = nn.Linear(in_dim, dh['units'])

        def fn(di):
            return {'Out': Dense(di['In'])}

        return fn, [Dense]

    return siso_pytorch_module('Dense', compile_fn, {'units': h_units})


def nonlinearity(h_nonlin_name):

    def compile_fn(di, dh):

        def fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = F.relu(di['In'])
            elif nonlin_name == 'tanh':
                Out = nn.Tanh()(di['In'])
            elif nonlin_name == 'elu':
                Out = F.elu(di['In'])
            else:
                raise ValueError
            return {"Out": Out}

        return fn, []

    return siso_pytorch_module('Nonlinearity', compile_fn,
                               {'nonlin_name': h_nonlin_name})


def dropout(h_keep_prob):

    def compile_fn(di, dh):
        Dropout = nn.Dropout(p=dh['keep_prob'])

        def fn(di):
            return {'Out': Dropout(di['In'])}

        return fn, [Dropout]

    return siso_pytorch_module('Dropout', compile_fn,
                               {'keep_prob': h_keep_prob})


def batch_normalization():

    def compile_fn(di, dh):
        (_, L) = di['In'].size()
        bn = nn.BatchNorm1d(L)

        def fn(di):
            return {'Out': bn(di['In'])}

        return fn, [bn]

    return siso_pytorch_module('BatchNormalization', compile_fn, {})


def dnn_net_simple(num_classes):

    # defining hyperparameter
    h_num_hidden = D(
        [64, 128, 256, 512,
         1024])  # number of hidden units for affine transform module
    h_nonlin_name = D(['relu', 'tanh',
                       'elu'])  # nonlinearity function names to choose from
    h_opt_drop = D(
        [0, 1])  # dropout optional hyperparameter; 0 is exclude, 1 is include
    h_drop_keep_prob = D([0.25, 0.5,
                          0.75])  # dropout probability to choose from
    h_opt_bn = D([0, 1])  # batch_norm optional hyperparameter
    h_swap = D([0, 1])  # order of swapping for permutation
    h_num_repeats = D([1, 2])  # 1 is appearing once, 2 is appearing twice

    # defining search space topology
    model = mo.siso_sequential([
        flatten(),
        mo.siso_repeat(
            lambda: mo.siso_sequential([
                dense(h_num_hidden),
                nonlinearity(h_nonlin_name),
                mo.siso_permutation([
                    lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob),
                                             h_opt_drop),
                    lambda: mo.siso_optional(batch_normalization, h_opt_bn),
                ], h_swap)
            ]), h_num_repeats),
        dense(D([num_classes]))
    ])

    return model


def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
             h_drop_keep_prob):
    return mo.siso_sequential([
        dense(h_num_hidden),
        nonlinearity(h_nonlin_name),
        mo.siso_permutation([
            lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob),
                                     h_opt_drop),
            lambda: mo.siso_optional(batch_normalization, h_opt_bn),
        ], h_swap)
    ])


def dnn_net(num_classes):
    h_nonlin_name = D(['relu', 'tanh', 'elu'])
    h_swap = D([0, 1])
    h_opt_drop = D([0, 1])
    h_opt_bn = D([0, 1])
    return mo.siso_sequential([
        flatten(),
        mo.siso_repeat(
            lambda: dnn_cell(
                D([64, 128, 256, 512, 1024]), h_nonlin_name, h_swap, h_opt_drop,
                h_opt_bn, D([0.25, 0.5, 0.75])), D([1, 2])),
        dense(D([num_classes]))
    ])


# Main/Searcher
import deep_architect.searchers.random as se
import deep_architect.core as co
import torchvision


def main():

    num_classes = 10
    num_samples = 3  # number of architecture to sample
    batch_size = 256
    best_val_acc, best_architecture = 0., -1

    # load and normalize data
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            './tmp/data/',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size,
        shuffle=True)
    # NOTE: using test as validation here, for simplicity sake.
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            './tmp/data/',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size,
        shuffle=True)

    # defining evaluator and searcher
    evaluator = SimpleClassifierEvaluator(
        train_loader,
        val_loader,
        num_classes,
        max_num_training_epochs=5,
        batch_size=batch_size,
        log_output_to_terminal=True)  # defining evaluator
    search_space_fn = mo.SearchSpaceFactory(lambda: dnn_net_simple(
        num_classes)).get_search_space
    searcher = se.RandomSearcher(search_space_fn)

    for i in xrange(num_samples):
        print("Sampling architecture %d" % i)
        inputs, outputs, _, searcher_eval_token = searcher.sample()
        val_acc = evaluator.evaluate(
            inputs,
            outputs)['val_acc']  # evaluate and return validation accuracy
        print("Finished evaluating architecture %d, validation accuracy is %f" %
              (i, val_acc))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_architecture = i
        searcher.update(val_acc, searcher_eval_token)
    print("Best validation accuracy is %f with architecture %d" %
          (best_val_acc, best_architecture))


# Evaluator
import torch.optim as optim
from deep_architect.helpers.pytorch import PyTorchModel, parameters


class SimpleClassifierEvaluator:

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 num_classes,
                 max_num_training_epochs=10,
                 batch_size=256,
                 learning_rate=1e-3,
                 display_step=1,
                 log_output_to_terminal=True):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes

        self.max_num_training_epochs = max_num_training_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_output_to_terminal = log_output_to_terminal
        self.display_step = display_step

    def compute_accuracy(self, inputs):
        self.network.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.val_dataset:
                output = self.network({'In': data})
                probs = F.softmax(output['Out'], dim=1)
                pred = probs.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()
        return (1.0 * correct / len(self.val_dataset.dataset))

    def evaluate(self, inputs, outputs):

        self.network = PyTorchModel(inputs, outputs)
        print self.batch_size, self.train_dataset.dataset[0][0].size()
        X = torch.ones(
            *(self.batch_size,) + self.train_dataset.dataset[0][0].size())
        self.network.forward({'In': X})  # to extract parameters
        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        for epoch in range(self.max_num_training_epochs):
            self.network.train()
            for batch_idx, (data, target) in enumerate(self.train_dataset):
                print epoch, batch_idx
                optimizer.zero_grad()
                output = self.network({'In': data})
                probs = F.softmax(output['Out'], dim=1)
                loss = F.nll_loss(probs, target)
                loss.backward()
                optimizer.step()

            val_acc = self.compute_accuracy(inputs)

            if self.log_output_to_terminal and epoch % self.display_step == 0:
                print('epoch: %d' % (epoch + 1),
                      'train loss: %.6f' % loss.item(),
                      'validation_accuracy: %.5f' % val_acc)

        val_acc = self.compute_accuracy(inputs)
        return {'val_acc': val_acc}


if __name__ == "__main__":
    main()