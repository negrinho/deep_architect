import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision
from torchvision.transforms import Compose, ToTensor, Lambda
import torch.optim as optim
import numpy as np

import deep_architect.core as co
import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.visualization as vi
import deep_architect.helpers.pytorch_support as hpt
import deep_architect.searchers.random as se
from deep_architect.helpers.pytorch_support import PyTorchModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dense(co.Module):

    def __init__(self, h_units):
        super().__init__(["in"], ["out"], {"units": h_units})

    def compile(self):
        units = self.hyperps["units"].get_value()
        x = self.inputs["in"].val
        self.m = nn.Linear(x.size(1), units)

    def forward(self):
        self.outputs["out"].val = self.m(self.inputs["in"].val)


class Nonlinearity(co.Module):

    def __init__(self, h_nonlin_name):
        super().__init__(["in"], ["out"], {"nonlin_name": h_nonlin_name})

    def compile(self):
        nonlin_name = self.hyperps["nonlin_name"].val
        if nonlin_name == 'relu':
            self.m = nn.ReLU()
        elif nonlin_name == 'tanh':
            self.m = nn.Tanh()
        elif nonlin_name == 'elu':
            self.m = nn.ELU()
        else:
            raise ValueError

    def forward(self):
        self.outputs["out"].val = self.m(self.inputs["in"].val)


class Dropout(co.Module):

    def __init__(self, h_drop_rate):
        super().__init__(["in"], ["out"], {"drop_rate": h_drop_rate})

    def compile(self):
        self.hyperps["drop_rate"].val
        self.m = nn.Dropout()

    def forward(self):
        self.outputs["out"].val = self.m(self.inputs["in"].val)


class BatchNormalization(co.Module):

    def __init__(self):
        super().__init__(["in"], ["out"], {})

    def compile(self):
        in_features = self.inputs['in'].val.size(1)
        self.m = nn.BatchNorm1d(in_features)

    def forward(self):
        self.outputs["out"].val = self.m(self.inputs["in"].val)


def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
             h_drop_rate):
    return mo.siso_sequential([
        Dense(h_num_hidden),
        Nonlinearity(h_nonlin_name),
        mo.SISOPermutation([
            lambda: mo.SISOOptional(lambda: Dropout(h_drop_rate), h_opt_drop),
            lambda: mo.SISOOptional(lambda: BatchNormalization(), h_opt_bn),
        ], h_swap)
    ])


def dnn_net(num_classes):
    h_nonlin_name = hp.Discrete(['relu', 'tanh', 'elu'])
    h_swap = hp.Discrete([0, 1])
    h_opt_drop = hp.Discrete([0, 1])
    h_opt_bn = hp.Discrete([0, 1])
    return mo.siso_sequential([
        mo.SISORepeat(
            lambda: dnn_cell(hp.Discrete([64, 128, 256, 512, 1024]),
                             h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
                             hp.Discrete([0.25, 0.5, 0.75])),
            hp.Discrete([1, 2, 4])),
        Dense(hp.Discrete([num_classes]))
    ])


class SimpleClassifierEvaluator:

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 num_training_epochs,
                 batch_size=256,
                 learning_rate=1e-4,
                 display_step=1,
                 log_output_to_terminal=True):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_data = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True)
        self.val_data = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size)
        self.num_training_epochs = num_training_epochs
        self.learning_rate = learning_rate
        self.log_output_to_terminal = log_output_to_terminal
        self.display_step = display_step

    def evaluate(self, inputs, outputs):
        init_data, _ = next(iter(self.train_data))
        model = hpt.PyTorchModel(inputs, outputs, {'in': init_data})
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        model.train()
        for epoch in range(self.num_training_epochs):
            for data, target in self.train_data:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model({'in': data})
                loss = F.cross_entropy(output["out"], target)
                loss.backward()
                optimizer.step()

            if self.log_output_to_terminal and epoch % self.display_step == 0:
                print('epoch: %d' % (epoch + 1),
                      'train loss: %.6f' % loss.item())

        # compute validation accuracy
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.val_data:
                output = model({'in': data})
                pred = output["out"].max(1)[1]
                correct += pred.eq(target).sum().item()
        val_acc = float(correct) / len(self.val_dataset)
        print("validation accuracy: %0.4f" % val_acc)

        return {'validation_accuracy': val_acc}


def main():
    num_classes = 10
    num_samples = 3
    num_training_epochs = 2
    batch_size = 256
    # NOTE: change to True for graph visualization
    show_graph = False

    data_transform = Compose([
        ToTensor(),
        Lambda(lambda x: x.reshape(-1)),
    ])

    trainval_dataset = torchvision.datasets.MNIST('./data',
                                                  train=True,
                                                  download=True,
                                                  transform=data_transform)
    train_dataset = torch.utils.data.Subset(trainval_dataset,
                                            np.arange(0, 50000))
    val_dataset = torch.utils.data.Subset(trainval_dataset,
                                          np.arange(50000, 60000))
    test_dataset = torchvision.datasets.MNIST('./data',
                                              train=False,
                                              download=True,
                                              transform=data_transform)

    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # defining evaluator and searcher
    evaluator = SimpleClassifierEvaluator(
        train_dataset,
        val_dataset,
        num_training_epochs=num_training_epochs,
        log_output_to_terminal=True)
    search_space_fn = lambda: dnn_net(num_classes)
    searcher = se.RandomSearcher(search_space_fn)

    for i in range(num_samples):
        inputs, outputs, _, searcher_eval_token = searcher.sample()
        if show_graph:
            # try setting draw_module_hyperparameter_info=False and
            # draw_hyperparameters=True for a different visualization.
            vi.draw_graph(outputs,
                          draw_module_hyperparameter_info=False,
                          draw_hyperparameters=True)

        results = evaluator.evaluate(inputs, outputs)
        searcher.update(results['validation_accuracy'], searcher_eval_token)


if __name__ == "__main__":
    main()