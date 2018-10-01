# Test with mnist
# Using TensorFlow Backend

# Search Space
import keras
import numpy as np

import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
from deep_architect.contrib.misc..search_spaces.tensorflow.common import siso_tfm

D = hp.Discrete # Discrete Hyperparameter

def flatten():
    def compile_fn(di, dh):
        Flatten = keras.layers.Flatten()
        def fn(di):
            return {'Out': Flatten(di['In'])}
        return fn
    return siso_tfm('Flatten', compile_fn, {}) # use siso_tfm for now

def dense(h_units):
    def compile_fn(di, dh):
        Dense = keras.layers.Dense(dh['units'])
        def fn(di):
            return {'Out' : Dense(di['In'])}
        return fn
    return siso_tfm('Dense', compile_fn, {'units' : h_units})

def nonlinearity(h_nonlin_name):
    def compile_fn(di, dh):
        def fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = keras.layers.Activation('relu')(di['In'])
            elif nonlin_name == 'tanh':
                Out = keras.layers.Activation('tanh')(di['In'])
            elif nonlin_name == 'elu':
                Out = keras.layers.Activation('elu')(di['In'])
            else:
                raise ValueError
            return {"Out" : Out}
        return fn
    return siso_tfm('Nonlinearity', compile_fn, {'nonlin_name' : h_nonlin_name})

def dropout(h_keep_prob):
    def compile_fn(di, dh):
        Dropout = keras.layers.Dropout(dh['keep_prob'])
        def fn(di):
            return {'Out' : Dropout(di['In'])}
        return fn
    return siso_tfm('Dropout', compile_fn, {'keep_prob' : h_keep_prob})

def batch_normalization():
    def compile_fn(di, dh):
        bn = keras.layers.BatchNormalization()
        def fn(di):
            return {'Out' : bn(di['In'])}
        return fn
    return siso_tfm('BatchNormalization', compile_fn, {})

def dnn_net_simple(num_classes):

        # defining hyperparameter
        h_num_hidden = D([64, 128, 256, 512, 1024]) # number of hidden units for affine transform module
        h_nonlin_name = D(['relu', 'tanh', 'elu']) # nonlinearity function names to choose from
        h_opt_drop = D([0, 1]) # dropout optional hyperparameter; 0 is exclude, 1 is include
        h_drop_keep_prob = D([0.25, 0.5, 0.75]) # dropout probability to choose from
        h_opt_bn = D([0, 1]) # batch_norm optional hyperparameter
        h_swap = D([0, 1]) # order of swapping for permutation
        h_num_repeats = D([1, 2]) # 1 is appearing once, 2 is appearing twice

        # defining search space topology
        model = mo.siso_sequential([
            flatten(),
            mo.siso_repeat(lambda: mo.siso_sequential([
                dense(h_num_hidden),
                nonlinearity(h_nonlin_name),
                mo.siso_permutation([
                    lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),
                    lambda: mo.siso_optional(batch_normalization, h_opt_bn),
                ], h_swap)
            ]), h_num_repeats),
            dense(D([num_classes]))
        ])

        return model



def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn, h_drop_keep_prob):
    return mo.siso_sequential([
        dense(h_num_hidden),
        nonlinearity(h_nonlin_name),
        mo.siso_permutation([
            lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),
            lambda: mo.siso_optional(batch_normalization, h_opt_bn),
        ], h_swap)])

def dnn_net(num_classes):
    h_nonlin_name = D(['relu', 'tanh', 'elu'])
    h_swap = D([0, 1])
    h_opt_drop = D([0, 1])
    h_opt_bn = D([0, 1])
    return mo.siso_sequential([
        flatten(),
        mo.siso_repeat(lambda: dnn_cell(
            D([64, 128, 256, 512, 1024]),
            h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
            D([0.25, 0.5, 0.75])), D([1, 2])),
        dense(D([num_classes]))])

# Evaluator

class SimpleClassifierEvaluator:

    def __init__(self, train_dataset, num_classes, batch_size=256,
                learning_rate=1e-3, metric='val_loss', resource_type='epoch'):

        self.train_dataset = train_dataset
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.val_split = 0.1 # 10% of dataset for validation
        self.metric = metric
        self.resource_type = resource_type

    def evaluate(self, inputs, outputs, resource):
        keras.backend.clear_session()

        (x_train, y_train) = self.train_dataset

        X = keras.layers.Input(x_train[0].shape)
        co.forward({inputs['In'] : X})
        logits = outputs['Out'].val
        probs = keras.layers.Softmax()(logits)
        model = keras.models.Model(inputs=[inputs['In'].val], outputs=[probs])
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        model.summary()
        history = model.fit(x_train, y_train,
                batch_size=self.batch_size,
                epochs=resource,
                validation_split=self.val_split)
        final_val_acc = history.history['val_acc'][-1]
        metric_values = history.history['val_acc'] if self.metric == 'val_accuracy' else history.history['loss']
        info = {self.resource_type: [i for i in range(resource)],
                self.metric: metric_values}
        results = {'val_accuracy': final_val_acc,
                   'history': info}
        return results

# Main/Searcher
import deep_architect.searchers.random as se
import deep_architect.core as co
from keras.datasets import mnist
from hyperband import SimpleArchitectureSearchHyperBand

def get_search_space(num_classes):
    def fn():
        co.Scope.reset_default_scope()
        inputs, outputs = dnn_net(num_classes)
        return inputs, outputs, {}
    return fn

def main():

    num_classes = 10
    num_samples = 3 # number of architecture to sample
    metric = 'val_accuracy' # evaluation metric
    resource_type = 'epoch'
    max_resource = 81 # max resource that a configuration can have

    # load and normalize data
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # defining searcher and evaluator
    evaluator = SimpleClassifierEvaluator((x_train, y_train), num_classes,
                                        max_num_training_epochs=5)
    searcher = se.RandomSearcher(get_search_space(num_classes))
    hyperband = SimpleArchitectureSearchHyperBand(searcher, hyperband, metric, resource_type)
    (best_config, best_perf) = hyperband.evaluate(max_resource)
    print("Best %s is %f with architecture %d" % (metric, best_perf[0], best_config[0]))

if __name__ == "__main__":
    main()