import tensorflow as tf
import numpy as np

import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
from deep_architect.contrib.misc.search_spaces.tensorflow.common import siso_tfm

D = hp.Discrete  # Discrete Hyperparameter


def dense(h_units):

    def compile_fn(di, dh):  # compile function
        Dense = tf.keras.layers.Dense(dh['units'])

        def fn(di):  # forward function
            return {'Out': Dense(di['In'])}

        return fn

    return siso_tfm('Dense', compile_fn, {'units': h_units})


def flatten():

    def compile_fn(di, dh):
        Flatten = tf.keras.layers.Flatten()

        def fn(di):
            return {'Out': Flatten(di['In'])}

        return fn

    return siso_tfm('Flatten', compile_fn, {})


def nonlinearity(h_nonlin_name):

    def compile_fn(di, dh):

        def fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = tf.keras.layers.Activation('relu')(di['In'])
            elif nonlin_name == 'tanh':
                Out = tf.keras.layers.Activation('tanh')(di['In'])
            elif nonlin_name == 'elu':
                Out = tf.keras.layers.Activation('elu')(di['In'])
            else:
                raise ValueError
            return {"Out": Out}

        return fn

    return siso_tfm('Nonlinearity', compile_fn, {'nonlin_name': h_nonlin_name})


def dropout(h_keep_prob):

    def compile_fn(di, dh):
        Dropout = tf.keras.layers.Dropout(dh['keep_prob'])

        def fn(di):
            return {'Out': Dropout(di['In'])}

        return fn

    return siso_tfm('Dropout', compile_fn, {'keep_prob': h_keep_prob})


def batch_normalization():

    def compile_fn(di, dh):
        bn = tf.keras.layers.BatchNormalization()

        def fn(di):
            return {'Out': bn(di['In'])}

        return fn

    return siso_tfm('BatchNormalization', compile_fn, {})


def dnn_net_simple(num_classes):

    # defining hyperparameter
    h_num_hidden = D([64, 128, 256, 512,
                      1024])  # number of hidden units for dense module
    h_nonlin_name = D(['relu', 'tanh',
                       'elu'])  # nonlinearity function names to choose from
    h_opt_drop = D(
        [0, 1])  # dropout optional hyperparameter; 0 is exclude, 1 is include
    h_drop_keep_prob = D([0.25, 0.5,
                          0.75])  # dropout probability to choose from
    h_opt_bn = D([0, 1])  # batch_norm optional hyperparameter
    h_perm = D([0, 1])  # order of swapping for permutation
    h_num_repeats = D([1, 2])  # 1 is appearing once, 2 is appearing twice

    # defining search space topology
    model = mo.siso_sequential([
        flatten(),
        mo.siso_repeat(lambda: mo.siso_sequential([
            dense(h_num_hidden),
            nonlinearity(h_nonlin_name),
            mo.siso_permutation([
                lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),
                lambda: mo.siso_optional(batch_normalization, h_opt_bn),
            ], h_perm)
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
            lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),
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
        mo.siso_repeat(lambda: dnn_cell(
            D([64, 128, 256, 512, 1024]),
            h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
            D([0.25, 0.5, 0.75])), D([1, 2])),
        dense(D([num_classes]))])


import deep_architect.searchers.random as se
import deep_architect.core as co


def get_search_space(num_classes):

    def fn():
        co.Scope.reset_default_scope()
        inputs, outputs = dnn_net(num_classes)
        return inputs, outputs, {}

    return fn


class SimpleClassifierEvaluator:

    def __init__(self,
                 train_dataset,
                 num_classes,
                 max_num_training_epochs=20,
                 batch_size=256,
                 learning_rate=1e-3):

        self.train_dataset = train_dataset
        self.num_classes = num_classes
        self.max_num_training_epochs = max_num_training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.val_split = 0.1  # 10% of dataset for validation

    def evaluate(self, inputs, outputs):
        tf.keras.backend.clear_session()
        tf.reset_default_graph()

        (x_train, y_train) = self.train_dataset

        X = tf.keras.layers.Input(x_train[0].shape)
        co.forward({inputs['In']: X})
        logits = outputs['Out'].val
        probs = tf.keras.layers.Softmax()(logits)
        model = tf.keras.models.Model(
            inputs=[inputs['In'].val], outputs=[probs])
        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        model.summary()
        history = model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.max_num_training_epochs,
            validation_split=self.val_split)

        results = {'val_acc': history.history['val_acc'][-1]}
        return results


def main():

    num_classes = 10
    num_samples = 3  # number of architecture to sample
    best_val_acc, best_architecture = 0., -1

    # load and normalize data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # defining evaluator and searcher
    evaluator = SimpleClassifierEvaluator((x_train, y_train),
                                          num_classes,
                                          max_num_training_epochs=5)
    searcher = se.RandomSearcher(get_search_space(num_classes))

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


if __name__ == "__main__":
    main()
