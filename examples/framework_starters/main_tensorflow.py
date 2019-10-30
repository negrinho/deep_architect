import tensorflow as tf
import numpy as np

import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.helpers.tensorflow_support as htf
import deep_architect.visualization as vi

from deep_architect.contrib.misc.datasets.loaders import load_mnist
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset
import deep_architect.searchers.random as se
import deep_architect.core as co

tf.logging.set_verbosity(tf.logging.ERROR)

D = hp.Discrete


def dense(h_units):
    return htf.siso_tensorflow_module_from_tensorflow_op_fn(
        tf.layers.Dense, {'units': h_units})


def dropout(h_drop_rate):

    def compile_fn(di, dh):
        p = tf.placeholder(tf.float32)

        def fn(di):
            return {'out': tf.nn.dropout(di['in'], p)}

        return fn, {p: 1.0 - dh['drop_rate']}, {p: 1.0}

    return htf.siso_tensorflow_module('Dropout', compile_fn,
                                      {'drop_rate': h_drop_rate})


# TODO: this needs to be fixed because it is not possible to do different
# forwards.
def batch_normalization():

    def compile_fn(di, dh):
        p_var = tf.placeholder(tf.bool)
        bn = tf.layers.BatchNormalization()

        def fn(di):
            return {'out': bn(di['in'], training=p_var)}

        return fn, {p_var: 1}, {p_var: 0}

    return htf.siso_tensorflow_module('BatchNormalization', compile_fn, {})


def nonlinearity(h_nonlin_name):

    def compile_fn(di, dh):

        def fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = tf.nn.relu(di['in'])
            elif nonlin_name == 'tanh':
                Out = tf.nn.tanh(di['in'])
            elif nonlin_name == 'elu':
                Out = tf.nn.elu(di['in'])
            else:
                raise ValueError
            return {"out": Out}

        return fn

    return htf.siso_tensorflow_module('Nonlinearity', compile_fn,
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
                 train_dataset,
                 val_dataset,
                 num_classes,
                 num_training_epochs,
                 batch_size=256,
                 learning_rate=1e-3,
                 display_step=1,
                 log_output_to_terminal=True):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.in_dim = list(train_dataset.next_batch(1)[0].shape[1:])

        self.num_training_epochs = num_training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.log_output_to_terminal = log_output_to_terminal
        self.display_step = display_step

    def compute_accuracy(self, sess, X_pl, y_pl, num_correct, dataset,
                         eval_feed):
        nc = 0
        num_left = dataset.get_num_examples()
        while num_left > 0:
            X_batch, y_batch = dataset.next_batch(self.batch_size)
            eval_feed.update({X_pl: X_batch, y_pl: y_batch})
            nc += sess.run(num_correct, feed_dict=eval_feed)
            # update the number of examples left.
            eff_batch_size = y_batch.shape[0]
            num_left -= eff_batch_size
        acc = float(nc) / dataset.get_num_examples()
        return acc

    def evaluate(self, inputs, outputs):
        tf.reset_default_graph()

        X_pl = tf.placeholder("float", [None] + self.in_dim)
        y_pl = tf.placeholder("float", [None, self.num_classes])
        lr_pl = tf.placeholder("float")
        co.forward({inputs['in']: X_pl})
        logits = outputs['out'].val
        train_feed, eval_feed = htf.get_feed_dicts(outputs)

        # define loss and optimizer
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_pl))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_pl)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = optimizer.minimize(loss)

        # for computing the accuracy of the model
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_pl, 1))
        num_correct = tf.reduce_sum(tf.cast(correct_prediction, "float"))

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            num_batches = int(self.train_dataset.get_num_examples() /
                              self.batch_size)
            for epoch in range(self.num_training_epochs):
                avg_loss = 0.
                for _ in range(num_batches):
                    X_batch, y_batch = self.train_dataset.next_batch(
                        self.batch_size)
                    train_feed.update({
                        X_pl: X_batch,
                        y_pl: y_batch,
                        lr_pl: self.learning_rate
                    })

                    _, c = sess.run([optimizer, loss], feed_dict=train_feed)
                    avg_loss += c / num_batches

                if self.log_output_to_terminal and epoch % self.display_step == 0:
                    print("epoch:", '%d' % (epoch + 1), "loss:",
                          "{:.9f}".format(avg_loss))

            val_acc = self.compute_accuracy(sess, X_pl, y_pl, num_correct,
                                            self.val_dataset, eval_feed)
            print("validation accuracy: %0.4f" % val_acc)
            results = {
                'validation_accuracy': val_acc,
                'num_parameters': htf.get_num_trainable_parameters()
            }

        return results


def main():
    num_classes = 10
    num_samples = 4
    num_training_epochs = 2
    # NOTE: change to True for graph visualization
    show_graph = False

    # load data
    (X_train, y_train, X_val, y_val, X_test, y_test) = load_mnist(flatten=True)
    train_dataset = InMemoryDataset(X_train, y_train, True)
    val_dataset = InMemoryDataset(X_val, y_val, False)
    test_dataset = InMemoryDataset(X_test, y_test, False)

    # defining evaluator and searcher
    evaluator = SimpleClassifierEvaluator(
        train_dataset,
        val_dataset,
        num_classes,
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
        # updating the searcher. no-op for the random searcher.
        searcher.update(results['validation_accuracy'], searcher_eval_token)


if __name__ == "__main__":
    main()