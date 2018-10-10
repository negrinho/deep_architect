# Search Space for DyNet
# NOTE: No Batch_norm since DyNet has not supported batch norm

import dynet as dy
import numpy as np

from deep_architect.helpers.dynet import DyParameterCollection, siso_dym
import deep_architect.modules as mo
import deep_architect.hyperparameters as hp

M = DyParameterCollection()
D = hp.Discrete


def flatten():

    def compile_fn(di, dh):
        shape = di['In'].dim()
        n = np.product(shape[0])
        Flatten = dy.reshape

        def fn(di):
            return {'Out': Flatten(di['In'], (n,))}

        return fn

    return siso_dym('Flatten', compile_fn, {})


def dense(h_u):

    def compile_fn(di, dh):
        shape = di['In'].dim()  # ((r, c), batch_dim)
        m, n = dh['units'], shape[0][0]
        pW = M.get_collection().add_parameters((m, n))
        pb = M.get_collection().add_parameters((m, 1))
        Dense = dy.affine_transform

        def fn(di):
            In = di['In']
            W, b = pW.expr(), pb.expr()
            # return {'Out': W*In + b}
            return {'Out': Dense([b, W, In])}

        return fn

    return siso_dym('Dense', compile_fn, {'units': h_u})


# just put here to streamline everything
def nonlinearity(h_nonlin_name):

    def compile_fn(di, dh):

        def fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = dy.rectify(di['In'])
            elif nonlin_name == 'elu':
                Out = dy.elu(di['In'])
            elif nonlin_name == 'tanh':
                Out = dy.tanh(di['In'])
            else:
                raise ValueError
            return {'Out': Out}

        return fn

    return siso_dym('Nonlinearity', compile_fn, {'nonlin_name': h_nonlin_name})


def dropout(h_keep_prob):

    def compile_fn(di, dh):
        p = dh['keep_prop']
        Dropout = dy.dropout

        def fn(di):
            return {'Out': Dropout(di['In'], p)}

        return fn

    return siso_dym('Dropout', compile_fn, {'keep_prop': h_keep_prob})


def dnn_net_simple(num_classes):

    # declaring hyperparameter
    h_nonlin_name = D(['relu', 'tanh',
                       'elu'])  # nonlinearity function names to choose from
    h_opt_drop = D(
        [0, 1])  # dropout optional hyperparameter; 0 is exclude, 1 is include
    h_drop_keep_prob = D([0.25, 0.5,
                          0.75])  # dropout probability to choose from
    h_num_hidden = D(
        [64, 128, 256, 512,
         1024])  # number of hidden units for affine transform module
    h_num_repeats = D([1, 2])  # 1 is appearing once, 2 is appearing twice

    # defining search space topology
    model = mo.siso_sequential([
        flatten(),
        mo.siso_repeat(
            lambda: mo.siso_sequential([
                dense(h_num_hidden),
                nonlinearity(h_nonlin_name),
                mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),]),
            h_num_repeats),
        dense(D([num_classes]))
    ])

    return model


def dnn_cell(h_num_hidden, h_nonlin_name, h_opt_drop, h_drop_keep_prob):
    return mo.siso_sequential([
        dense(h_num_hidden),
        nonlinearity(h_nonlin_name),
        mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop)
    ])


def dnn_net(num_classes):
    h_nonlin_name = D(['relu', 'tanh', 'elu'])
    h_opt_drop = D([0, 1])
    return mo.siso_sequential([
        flatten(),
        mo.siso_repeat(lambda: dnn_cell(
            D([64, 128, 256, 512, 1024]),
            h_nonlin_name, h_opt_drop,
            D([0.25, 0.5, 0.75])), D([1, 2])),
        dense(D([num_classes]))])


# Main/Searcher
# Getting and reading mnist data adapted from here:
# https://github.com/clab/dynet/blob/master/examples/mnist/mnist-autobatch.py
import deep_architect.searchers.random as se
import deep_architect.core as co
from deep_architect.contrib.misc.datasets.loaders import load_mnist


def get_search_space(num_classes):

    def fn():
        co.Scope.reset_default_scope()
        inputs, outputs = dnn_net(num_classes)
        return inputs, outputs, {}

    return fn


def main():

    num_classes = 10
    num_samples = 3  # number of architecture to sample
    best_val_acc, best_architecture = 0., -1

    # donwload and normalize data, using test as val for simplicity
    X_train, y_train, X_val, y_val, _, _ = load_mnist(
        'data/mnist', normalize_range=True)

    # defining evaluator
    evaluator = SimpleClassifierEvaluator((X_train, y_train), (X_val, y_val),
                                          num_classes,
                                          max_num_training_epochs=5,
                                          log_output_to_terminal=True)
    searcher = se.RandomSearcher(get_search_space(num_classes))
    for i in xrange(num_samples):
        print("Sampling architecture %d" % i)
        M.renew_collection()
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
import random


class SimpleClassifierEvaluator:

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 num_classes,
                 max_num_training_epochs=10,
                 batch_size=16,
                 learning_rate=1e-3,
                 display_step=1,
                 log_output_to_terminal=True):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.max_num_training_epochs = max_num_training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.log_output_to_terminal = log_output_to_terminal
        self.display_step = display_step

    def compute_accuracy(self, inputs, outputs):
        correct = 0
        for (label, img) in self.val_dataset:
            dy.renew_cg()
            x = dy.inputVector(img)
            co.forward({inputs['In']: x})
            logits = outputs['Out'].val
            pred = np.argmax(logits.npvalue())
            if (label == pred): correct += 1
        return (1.0 * correct / len(self.val_dataset))

    def evaluate(self, inputs, outputs):
        params = M.get_collection()
        optimizer = dy.SimpleSGDTrainer(params, self.learning_rate)
        num_batches = int(len(self.train_dataset) / self.batch_size)
        for epoch in range(self.max_num_training_epochs):
            random.shuffle(self.train_dataset)
            i = 0
            total_loss = 0
            while (i < len(self.train_dataset)):
                dy.renew_cg()
                mbsize = min(self.batch_size, len(self.train_dataset) - i)
                minibatch = self.train_dataset[i:i + mbsize]
                losses = []
                for (label, img) in minibatch:
                    x = dy.inputVector(img)
                    co.forward({inputs['In']: x})
                    logits = outputs['Out'].val
                    loss = dy.pickneglogsoftmax(logits, label)
                    losses.append(loss)
                mbloss = dy.esum(losses) / mbsize
                mbloss.backward()
                optimizer.update()
                total_loss += mbloss.scalar_value()
                i += mbsize

            val_acc = self.compute_accuracy(inputs, outputs)
            if self.log_output_to_terminal and epoch % self.display_step == 0:
                print("epoch:", '%d' % (epoch + 1), "loss:", "{:.9f}".format(
                    total_loss / num_batches), "validation_accuracy:",
                      "%.5f" % val_acc)

        val_acc = self.compute_accuracy(inputs, outputs)
        return {'val_acc': val_acc}


if __name__ == "__main__":
    main()