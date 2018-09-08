# Search Space for DyNet
# NOTE: No Batch_norm since DyNet has not supported batch norm 

import dynet as dy 
import numpy as np 

from dynet_tmp import DyParameterCollection, siso_dym
import deep_architect.modules as mo 
import deep_architect.hyperparameters as hp

M = DyParameterCollection() 
D = hp.Discrete

def Flatten(): 
    def cfn(di, dh): 
        shape = di['In'].dim()
        n = np.product(shape[0])
        def fn(di): 
            return {'Out': dy.reshape(di['In'], (n,))} 
        return fn 
    return siso_dym('Flatten', cfn, {})

def Dense(h_u):
    def cfn(di, dh):
        shape = di['In'].dim() # ((r, c), batch_dim)
        m, n = dh['units'], shape[0][0]
        pW = M.get_collection().add_parameters((m, n))
        pb = M.get_collection().add_parameters((m, 1))
        def fn(di): 
            In = di['In']
            W, b = pW.expr(), pb.expr()
            # return {'Out': W*In + b}
            return {'Out': dy.affine_transform([b, W, In])}
        return fn 
    return siso_dym('Dense', cfn, {'units': h_u})

# just put here to streamline everything 
def Nonlinearity(h_nonlin_name): 
    def cfn(di, dh): 
        def fn(di): 
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu': 
                Out = dy.rectify(di['In'])
            elif nonlin_name == 'elu': 
                Out = dy.elu(di['In'])
            elif nonlin_name == 'sigmoid': 
                Out = dy.logistic(di['In'])
            elif nonlin_name == 'tanh':
                Out = dy.tanh(di['In'])
            else: 
                raise ValueError
            return {'Out': Out}
        return fn 
    return siso_dym('Nonlinearity', cfn, {'nonlin_name' : h_nonlin_name})

def Dropout(h_keep_prob): 
    def cfn(di, dh): 
        p = dh['keep_prop']
        def fn(di): 
            return {'Out': dy.dropout(di['In'], p)}
        return fn 
    return siso_dym('Dropout', cfn, {'keep_prop': h_keep_prob})


def dnn_net_simple(num_classes): 

        # declaring hyperparameter
        h_nonlin_name = D(['relu', 'tanh', 'elu', 'softplus']) # nonlinearity function names to choose from
        h_opt_drop = D([0, 1]) # dropout optional hyperparameter; 0 is exclude, 1 is include 
        h_drop_keep_prob = D([0.25, 0.5, 0.75]) # dropout probability to choose from 
        h_num_hidden = D([64, 128, 256, 512, 1024]) # number of hidden units for affine transform module 
        h_num_repeats = D([1, 2]) # 1 is appearing once, 2 is appearing twice
        
        # defining search space topology 
        model = mo.siso_sequential([
            Flatten(), 
            mo.siso_repeat(lambda: mo.siso_sequential([
                Dense(h_num_hidden),
                Nonlinearity(h_nonlin_name),
                mo.siso_optional(lambda: Dropout(h_drop_keep_prob), h_opt_drop),
            ]), h_num_repeats),
            Dense(D([num_classes]))
        ])
        
        return model 

def dnn_cell(h_num_hidden, h_nonlin_name, h_opt_drop, h_drop_keep_prob):
    return mo.siso_sequential([
        Dense(h_num_hidden),
        Nonlinearity(h_nonlin_name),
        mo.siso_optional(lambda: Dropout(h_drop_keep_prob), h_opt_drop)
    ])

def dnn_net(num_classes):
    h_nonlin_name = D(['relu', 'tanh', 'elu', 'softplus'])
    h_opt_drop = D([0, 1])
    return mo.siso_sequential([
        Flatten(), 
        mo.siso_repeat(lambda: dnn_cell(
            D([64, 128, 256, 512, 1024]),
            h_nonlin_name, h_opt_drop,
            D([0.25, 0.5, 0.75])), D([1, 2])),
        Dense(D([num_classes]))])

# Main/Searcher
# Getting and reading mnist data adapted from here: 
# https://github.com/clab/dynet/blob/master/examples/mnist/mnist-autobatch.py 
import deep_architect.searchers.random as se
import deep_architect.core as co 
from deep_architect.contrib.useful.datasets.loaders import read_mnist, download_examples

def get_search_space(num_classes):
    def fn(): 
        co.Scope.reset_default_scope()
        inputs, outputs = dnn_net(num_classes)
        return inputs, outputs, {}
    return fn

def main():
    num_classes = 10
    num_samples = 3
    data_path = './tmp_dynet/data/'
    # download_examples(data_path)
    train_dataset = [(label, img) for (label, img) in read_mnist('training', data_path)]
    test_dataset = [(label, img) for (label, img) in read_mnist('testing', data_path)]
    evaluator = SimpleClassifierEvaluator(train_dataset, test_dataset, num_classes,
                './tmp', max_num_training_epochs=5, log_output_to_terminal=True) # defining evaluator 

    searcher = se.RandomSearcher(get_search_space(num_classes))
    best_val_acc, best_architecture = 0., -1
    for i in xrange(num_samples):
        print("Sampling architecture %d" % i)
        M.renew_collection()
        inputs, outputs, hs, _, searcher_eval_token = searcher.sample()
        val_acc = evaluator.evaluate(inputs, outputs, hs)['validation_accuracy'] # evaluate and return validation accuracy
        print("Finished evaluating architecture %d, validation accuracy is %f" % (i, val_acc))
        if val_acc > best_val_acc: 
            best_val_acc = val_acc
            best_architecture = i
        searcher.update(val_acc, searcher_eval_token)
    print("Best validation accuracy is %f with architecture %d" % (best_val_acc, best_architecture)) 

# Evaluator 
import random 

class SimpleClassifierEvaluator:

    def __init__(self, train_dataset, val_dataset, num_classes, model_path,
            max_num_training_epochs=10, optimizer_type='sgd', batch_size=16,
            learning_rate=1e-3, display_step=1, log_output_to_terminal=True, 
            test_dataset=None):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes

        self.max_num_training_epochs = max_num_training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.log_output_to_terminal = log_output_to_terminal
        self.display_step = display_step
        self.model_path = model_path
        self.test_dataset = test_dataset

    def get_optimizer(self): 
        optimizer = None 
        params = M.get_collection()
        if self.optimizer_type == 'adam':
            optimizer = dy.SimpleSGDTrainer(params, learning_rate=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            optimizer = dy.AdamTrainer(params, self.learning_rate)
        elif self.optimizer_type == 'sgd_mom':
            optimizer = MomentumSGDTrainer(params, learning_rate=self.learning_rate, momentum=0.99)
        else:
            raise ValueError("Unknown optimizer.")
        return optimizer 

    def compute_accuracy(self, inputs, outputs): 
        correct = 0 
        for (label, img) in self.val_dataset: 
            dy.renew_cg()
            x = dy.inputVector(img)
            co.forward({inputs['In'] : x})
            logits = outputs['Out'].val
            pred = np.argmax(logits.npvalue())
            if (label == pred): correct += 1
        return (1.0*correct / len(self.val_dataset))

    def evaluate(self, inputs, outputs, hs):

        optimizer = self.get_optimizer()
        num_batches = int(len(self.train_dataset) / self.batch_size) 
        for epoch in range(self.max_num_training_epochs): 
            random.shuffle(self.train_dataset)
            i = 0
            total_loss = 0 
            while (i < len(self.train_dataset)):
                dy.renew_cg()
                mbsize = min(self.batch_size, len(self.train_dataset)-i) 
                minibatch = self.train_dataset[i:i+mbsize]
                losses = []
                for (label, img) in minibatch:
                    x = dy.inputVector(img)
                    co.forward({inputs['In'] : x})
                    logits = outputs['Out'].val
                    loss = dy.pickneglogsoftmax(logits, label)
                    losses.append(loss)
                mbloss = dy.esum(losses)/mbsize 
                mbloss.backward()
                optimizer.update()
                total_loss += mbloss.scalar_value()
                i += mbsize

            val_acc = self.compute_accuracy(inputs, outputs)
            if self.log_output_to_terminal and epoch % self.display_step == 0: 
                print("epoch:", '%d' % (epoch + 1),
                      "loss:", "{:.9f}".format(total_loss / num_batches),
                      "validation_accuracy:", "%.5f" % val_acc)
    
        
        val_acc = self.compute_accuracy(inputs, outputs)
        return {'validation_accuracy': val_acc}
        

if __name__ == "__main__": 
    main() 