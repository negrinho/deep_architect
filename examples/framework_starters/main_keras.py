import keras
import numpy as np

import deep_architect.core as co
import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.searchers.random as se
import deep_architect.helpers.keras_support as hke
import deep_architect.search_logging as sl
import deep_architect.visualization as vi

from keras.layers import Dropout, Activation, BatchNormalization, Dense, Input
from keras.models import Model
from keras.optimizers import Adam

from keras.datasets import mnist

D = hp.Discrete


def dense(h_units):
    return hke.siso_keras_module_from_keras_layer_fn(Dense, {'units': h_units})


def dropout(h_drop_rate):
    return hke.siso_keras_module_from_keras_layer_fn(Dropout,
                                                     {'rate': h_drop_rate})


def batch_normalization():
    return hke.siso_keras_module_from_keras_layer_fn(BatchNormalization, {})


def nonlinearity(h_nonlin_name):

    def compile_fn(di, dh):

        def fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = Activation('relu')(di['in'])
            elif nonlin_name == 'tanh':
                Out = Activation('tanh')(di['in'])
            elif nonlin_name == 'elu':
                Out = Activation('elu')(di['in'])
            else:
                raise ValueError
            return {"out": Out}

        return fn

    return hke.siso_keras_module('Nonlinearity', compile_fn,
                                 {'nonlin_name': h_nonlin_name})


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
            lambda: dnn_cell(D([64, 128, 256, 512, 1024]),
                             h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
                             D([0.25, 0.5, 0.75])), D([1, 2, 4])),
        dense(D([num_classes]))
    ])


class SimpleClassifierEvaluator:

    def __init__(self,
                 X_train,
                 y_train,
                 X_val,
                 y_val,
                 num_classes,
                 num_training_epochs,
                 batch_size=256,
                 learning_rate=1e-3):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_classes = num_classes
        self.num_training_epochs = num_training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def evaluate(self, inputs, outputs):
        keras.backend.clear_session()

        X = Input(self.X_train[0].shape)
        co.forward({inputs['in']: X})
        logits = outputs['out'].val
        probs = Activation('softmax')(logits)

        model = Model(inputs=[inputs['in'].val], outputs=[probs])
        model.compile(optimizer=Adam(lr=self.learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(self.X_train,
                            self.y_train,
                            batch_size=self.batch_size,
                            epochs=self.num_training_epochs,
                            validation_data=(self.X_val, self.y_val))
        results = {'validation_accuracy': history.history['val_accuracy'][-1]}
        return results


def main():
    num_classes = 10
    num_samples = 4
    num_training_epochs = 2
    validation_frac = 0.2
    # NOTE: change to True for graph visualization
    show_graph = False

    # load the data.
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    fn = lambda X: X.reshape((X.shape[0], -1))
    X_train = fn(X_train) / 255.0
    X_test = fn(X_test) / 255.0
    num_train = int((1.0 - validation_frac) * X_train.shape[0])
    X_train, X_val = X_train[:num_train], X_train[num_train:]
    y_train, y_val = y_train[:num_train], y_train[num_train:]

    # define the search and the evalutor
    evaluator = SimpleClassifierEvaluator(
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes,
        num_training_epochs=num_training_epochs)
    search_space_fn = lambda: dnn_net(num_classes)
    searcher = se.RandomSearcher(search_space_fn)

    for i in range(num_samples):
        (inputs, outputs, hyperp_value_lst,
         searcher_eval_token) = searcher.sample()
        if show_graph:
            # try setting draw_module_hyperparameter_info=False and
            # draw_hyperparameters=True for a different visualization.
            vi.draw_graph(outputs,
                          draw_module_hyperparameter_info=False,
                          draw_hyperparameters=True)
        results = evaluator.evaluate(inputs, outputs)
        # updating the searcher. no-op for the random searcher.
        searcher.update(results['validation_accuracy'], searcher_eval_token)


if __name__ == "__main__":
    main()