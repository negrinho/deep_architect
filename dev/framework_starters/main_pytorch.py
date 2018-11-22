from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.visualization as vi
import deep_architect.helpers.pytorch as hpt
import deep_architect.searchers.random as se
import torchvision

import torch.optim as optim
from deep_architect.helpers.pytorch import PyTorchModel

D = hp.Discrete


def dense(h_units):

    def compile_fn(di, dh):
        (_, in_features) = di['In'].size()
        m = nn.Linear(in_features, dh['units'])

        def fn(di):
            return {'Out': m(di['In'])}

        return fn, [m]

    return hpt.siso_pytorch_module('Dense', compile_fn, {'units': h_units})


def nonlinearity(h_nonlin_name):

    def Nonlinearity(nonlin_name):
        if nonlin_name == 'relu':
            m = nn.ReLU()
        elif nonlin_name == 'tanh':
            m = nn.Tanh()
        elif nonlin_name == 'elu':
            m = nn.ELU()
        else:
            raise ValueError

        return m

    return hpt.siso_pytorch_module_from_pytorch_layer_fn(
        Nonlinearity, {'nonlin_name': h_nonlin_name})


def dropout(h_drop_rate):
    return hpt.siso_pytorch_module_from_pytorch_layer_fn(
        nn.Dropout, {'p': h_drop_rate})


def batch_normalization():

    def compile_fn(di, dh):
        (_, in_features) = di['In'].size()
        bn = nn.BatchNorm1d(in_features)

        def fn(di):
            return {'Out': bn(di['In'])}

        return fn, [bn]

    return hpt.siso_pytorch_module('BatchNormalization', compile_fn, {})


def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
             h_drop_rate):
    return mo.siso_sequential([
        dense(h_num_hidden),
        nonlinearity(h_nonlin_name),
        mo.siso_permutation([
            lambda: mo.siso_optional(lambda: dropout(h_drop_rate), h_opt_drop),
            lambda: mo.siso_optional(batch_normalization, h_opt_bn),
        ], h_swap)
    ])


def dnn_net(num_classes):
    h_nonlin_name = D(['relu', 'tanh', 'elu'])
    h_swap = D([0, 1])
    h_opt_drop = D([0, 1])
    h_opt_bn = D([0, 1])
    return mo.siso_sequential([
        mo.siso_repeat(
            lambda: dnn_cell(
                D([64, 128, 256, 512, 1024]), h_nonlin_name, h_swap, h_opt_drop,
                h_opt_bn, D([0.25, 0.5, 0.75])), D([1, 2, 4])),
        dense(D([num_classes]))
    ])


class SimpleClassifierEvaluator:

    def __init__(self,
                 train_loader,
                 val_loader,
                 num_classes,
                 num_training_epochs,
                 batch_size=256,
                 learning_rate=1e-4,
                 display_step=1,
                 log_output_to_terminal=True):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.in_features = np.product(train_loader.dataset[0][0].size())

        self.num_training_epochs = num_training_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_output_to_terminal = log_output_to_terminal
        self.display_step = display_step

    def evaluate(self, inputs, outputs):
        network = hpt.PyTorchModel(inputs, outputs)
        # NOTE: instantiation of parameters requires passing data through the
        # model once.
        network.eval()
        network.forward({'In': torch.zeros(self.batch_size, self.in_features)})
        optimizer = optim.Adam(network.parameters(), lr=self.learning_rate)
        network.train()
        for epoch in range(self.num_training_epochs):
            for (data, target) in self.train_loader:
                optimizer.zero_grad()
                output = network({'In': data.view(-1, self.in_features)})
                loss = F.cross_entropy(output["Out"], target)
                loss.backward()
                optimizer.step()

            if self.log_output_to_terminal and epoch % self.display_step == 0:
                print('epoch: %d' % (epoch + 1),
                      'train loss: %.6f' % loss.item())

        # compute validation accuracy
        network.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                output = network({'In': data.view(-1, self.in_features)})
                pred = output["Out"].data.max(1)[1]
                correct += pred.eq(target.data).sum().item()
        val_acc = float(correct) / len(self.val_loader.dataset)
        print("validation accuracy: %0.4f" % val_acc)

        return {'validation_accuracy': val_acc}


def main():
    num_classes = 10
    num_samples = 3
    num_training_epochs = 2
    batch_size = 256
    # NOTE: change to True for graph visualization
    show_graph = False

    ### NOTE: revisit this.
    # load and normalize data
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            './tmp/data/',
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True)
    # NOTE: using test as validation here, for simplicity sake.
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            './tmp/data/',
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=False)

    # defining evaluator and searcher
    evaluator = SimpleClassifierEvaluator(
        train_loader,
        val_loader,
        num_classes,
        num_training_epochs=num_training_epochs,
        batch_size=batch_size,
        log_output_to_terminal=True)
    ssf = mo.SearchSpaceFactory(lambda: dnn_net(num_classes))
    searcher = se.RandomSearcher(ssf.get_search_space)

    for i in xrange(num_samples):
        inputs, outputs, _, searcher_eval_token = searcher.sample()
        if show_graph:
            # try setting draw_module_hyperparameter_info=False and
            # draw_hyperparameters=True for a different visualization.
            vi.draw_graph(
                outputs.values(),
                draw_module_hyperparameter_info=False,
                draw_hyperparameters=True)

        results = evaluator.evaluate(inputs, outputs)
        searcher.update(results['validation_accuracy'], searcher_eval_token)


if __name__ == "__main__":
    main()