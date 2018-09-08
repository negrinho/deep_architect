# Search Space 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

import deep_architect.modules as mo  
from deep_architect.contrib.useful.search_spaces.pytorch.common import siso_torchm, D

def Flatten(): 
    def cfn(di, dh): 
        shape = di['In'].size() 
        n = np.product(shape[1:])
        def fn(di): 
            return {'Out': (di['In']).view(-1, n)} 
        return fn, []
    return siso_torchm('Flatten', cfn, {})

def Dense(h_units): 
    def cfn(di, dh):
        (_, in_dim) = di['In'].size()
        dense = nn.Linear(in_dim, dh['units'])
        def fn(di):
            return {'Out' : dense(di['In'])}
        return fn, [dense]
    return siso_torchm('Dense', cfn, {'units' : h_units})

def Nonlinearity(h_nonlin_name):
    def cfn(di, dh):
        def fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = F.relu(di['In'])
            elif nonlin_name == 'tanh':
                Out = nn.Tanh()(di['In'])
            elif nonlin_name == 'elu':
                Out = F.elu(di['In'])
            elif nonlin_name == 'softplus':
                Out = F.softplus(di['In'])
            else:
                raise ValueError
            return {"Out" : Out}
        return fn, []
    return siso_torchm('Nonlinearity', cfn, {'nonlin_name' : h_nonlin_name})

def Dropout(h_keep_prob):
    def cfn(di, dh):
        dropout_layer = nn.Dropout(p=dh['keep_prob'])
        def fn(di):
            return {'Out' : dropout_layer(di['In'])}
        return fn, [dropout_layer]
    return siso_torchm('Dropout', cfn, {'keep_prob' : h_keep_prob})

def BatchNormalization():
    def cfn(di, dh):
        (_, L) = di['In'].size()
        batch_norm = nn.BatchNorm1d(L)
        def fn(di):
            return {'Out' : batch_norm(di['In'])}
        return fn, [batch_norm]
    return siso_torchm('BatchNormalization', cfn, {})

def dnn_net_simple(num_classes): 

        # declaring hyperparameter
        h_nonlin_name = D(['relu', 'tanh', 'elu', 'softplus']) # nonlinearity function names to choose from
        h_opt_drop = D([0, 1]) # dropout optional hyperparameter; 0 is exclude, 1 is include 
        h_drop_keep_prob = D([0.25, 0.5, 0.75]) # dropout probability to choose from 
        h_opt_bn = D([0, 1]) 
        h_num_hidden = D([64, 128, 256, 512, 1024]) # number of hidden units for affine transform module 
        h_swap = D([0, 1]) # order of swapping for permutation 
        h_num_repeats = D([1, 2]) # 1 is appearing once, 2 is appearing twice
        
        # defining search space topology 
        model = mo.siso_sequential([
            Flatten(),
            mo.siso_repeat(lambda: mo.siso_sequential([
                Dense(h_num_hidden),
                Nonlinearity(h_nonlin_name),
                mo.siso_permutation([
                    lambda: mo.siso_optional(lambda: Dropout(h_drop_keep_prob), h_opt_drop),
                    lambda: mo.siso_optional(BatchNormalization, h_opt_bn),
                ], h_swap)
            ]), h_num_repeats),
            Dense(D([num_classes]))
        ])
        
        return model 

def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn, h_drop_keep_prob):
    return mo.siso_sequential([
        Dense(h_num_hidden),
        Nonlinearity(h_nonlin_name),
        mo.siso_permutation([
            lambda: mo.siso_optional(lambda: Dropout(h_drop_keep_prob), h_opt_drop),
            lambda: mo.siso_optional(BatchNormalization, h_opt_bn),
        ], h_swap)])

def dnn_net(num_classes):
    h_nonlin_name = D(['relu', 'tanh', 'elu', 'softplus'])
    h_swap = D([0, 1])
    h_opt_drop = D([0, 1])
    h_opt_bn = D([0, 1])
    return mo.siso_sequential([
        Flatten(), 
        mo.siso_repeat(lambda: dnn_cell(
            D([64, 128, 256, 512, 1024]),
            h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
            D([0.25, 0.5, 0.75])), D([1, 2])),
        Dense(D([num_classes]))])

# Main/Searcher 
import deep_architect.searchers.random as se
import deep_architect.core as co 
import torchvision

def get_search_space(num_classes):
    def fn(): 
        co.Scope.reset_default_scope()
        inputs, outputs = dnn_net(num_classes)
        return inputs, outputs, {}
    return fn

def main():
    num_classes = 10
    num_samples = 3
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./tmp/data/', train=True, download=True, 
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(), 
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                    ])), 
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./tmp/data/', train=False, download=True, 
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(), 
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                    ])), 
        batch_size=batch_size, shuffle=True)
    evaluator = SimpleClassifierEvaluator(train_loader, test_loader, num_classes,
                    './tmp', max_num_training_epochs=5, batch_size=batch_size, 
                    log_output_to_terminal=True) # defining evaluator 

    searcher = se.RandomSearcher(get_search_space(num_classes))
    best_val_acc, best_architecture = 0., -1
    for i in xrange(num_samples):
        print("Sampling architecture %d" % i)
        inputs, outputs, hs, _, searcher_eval_token = searcher.sample()
        val_acc = evaluator.evaluate(inputs, outputs, hs)['validation_accuracy'] # evaluate and return validation accuracy
        print("Finished evaluating architecture %d, validation accuracy is %f" % (i, val_acc))
        if val_acc > best_val_acc: 
            best_val_acc = val_acc
            best_architecture = i
        searcher.update(val_acc, searcher_eval_token)
    print("Best validation accuracy is %f with architecture %d" % (best_val_acc, best_architecture)) 

# Evaluator 
import torch.optim as optim
from deep_architect.helpers.pytorch import PyTNetContainer, parameters

class SimpleClassifierEvaluator:

    def __init__(self, train_dataset, val_dataset, num_classes, model_path,
            max_num_training_epochs=10, batch_size=256, optimizer_type='adam',
            learning_rate=1e-3, display_step=1, log_output_to_terminal=True, 
            test_dataset=None):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes

        self.max_num_training_epochs = max_num_training_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.log_output_to_terminal = log_output_to_terminal
        self.display_step = display_step
        self.model_path = model_path
        self.test_dataset = test_dataset
        self.display_step = display_step

    def get_optimizer(self, params): 
        optimizer = None 
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            optimizer = optim.SGD(params, lr=self.learning_rate)
        elif self.optimizer_type == 'sgd_mom':
            optimizer = optim.SGD(params, lr=self.learning_rate, momentum=0.99)
        else:
            raise ValueError("Unknown optimizer.")
        return optimizer 
    
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

    def evaluate(self, inputs, outputs, hs):

        self.network = PyTNetContainer(inputs, outputs)
        X = torch.ones(()).new_empty((self.batch_size, ) + self.train_dataset.dataset[0][0].size())
        self.network.forward({'In': X}) # to extract parameters
        self.optimizer = self.get_optimizer(self.network.parameters())
        for epoch in range(self.max_num_training_epochs):
            self.network.train()
            for batch_idx, (data, target) in enumerate(self.train_dataset):
                self.optimizer.zero_grad()
                output = self.network({'In': data})
                probs = F.softmax(output['Out'], dim=1)
                loss = F.nll_loss(probs, target)
                loss.backward()
                self.optimizer.step()

            val_acc = self.compute_accuracy(inputs)
            print "val_acc: ", val_acc

            if self.log_output_to_terminal and epoch % self.display_step == 0:
                print('epoch: %d' % epoch, 
                      'train loss: %.6f' % loss.item(), 
                      'validation_accuracy: %.5f' % val_acc)

        val_acc = self.compute_accuracy(inputs)
        return {'validation_accuracy': val_acc}
        

       
        

if __name__ == "__main__": 
    main() 