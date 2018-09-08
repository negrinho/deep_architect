# Search Space 
from __future__ import print_function
import tensorflow as tf 
import numpy as np

import deep_architect.modules as mo  
from deep_architect.contrib.useful.search_spaces.tensorflow.common import siso_tfm, D

tf.logging.set_verbosity(tf.logging.ERROR)

def Flatten():
    def cfn(di, dh): 
        def fn(di): 
            return {'Out' : tf.layers.flatten(di['In'])}
        return fn 
    return siso_tfm('Flatten', cfn, {})

def Dense(h_units): 
    def cfn(di, dh):
        def fn(di):
            return {'Out' : tf.layers.dense(di['In'], dh['units'])}
        return fn
    return siso_tfm('Dense', cfn, {'units' : h_units})

def Nonlinearity(h_nonlin_name):
    def cfn(di, dh):
        def fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = tf.nn.relu(di['In'])
            elif nonlin_name == 'tanh':
                Out = tf.nn.tanh(di['In'])
            elif nonlin_name == 'elu':
                Out = tf.nn.elu(di['In'])
            elif nonlin_name == 'softplus':
                Out = tf.nn.softplus(di['In'])
            else:
                raise ValueError
            return {"Out" : Out}
        return fn
    return siso_tfm('Nonlinearity', cfn, {'nonlin_name' : h_nonlin_name})

def Dropout(h_keep_prob):
    def cfn(di, dh):
        p = tf.placeholder(tf.float32)
        def fn(di):
            return {'Out' : tf.nn.dropout(di['In'], p)}
        return fn, {p : dh['keep_prob']}, {p : 1.0}
    return siso_tfm('Dropout', cfn, {'keep_prob' : h_keep_prob})

def BatchNormalization():
    def cfn(di, dh):
        p_var = tf.placeholder(tf.bool)
        def fn(di):
            return {'Out' : tf.layers.batch_normalization(di['In'], training=p_var)}
        return fn, {p_var : 1}, {p_var : 0}
    return siso_tfm('BatchNormalization', cfn, {})

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
from deep_architect.contrib.useful.datasets.loaders import load_mnist
from deep_architect.contrib.useful.datasets.dataset import InMemoryDataset
import deep_architect.searchers.random as se
import deep_architect.core as co 

def get_search_space(num_classes):
    def fn(): 
        co.Scope.reset_default_scope()
        inputs, outputs = dnn_net(num_classes)
        return inputs, outputs, {}
    return fn

def main():
    num_classes = 10
    num_samples = 20
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_mnist('data/mnist')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
                './temp', max_num_training_epochs=5, log_output_to_terminal=True) # defining evaluator 

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
import deep_architect.helpers.tensorflow as htf

class SimpleClassifierEvaluator:

    def __init__(self, train_dataset, val_dataset, num_classes, model_path,
            max_num_training_epochs=10, optimizer_type='adam', batch_size=256,
            learning_rate=1e-3, display_step=1, log_output_to_terminal=True, 
            test_dataset=None):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.in_dim = list(train_dataset.next_batch(1)[0].shape[1:])

        self.max_num_training_epochs = max_num_training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.log_output_to_terminal = log_output_to_terminal
        self.display_step = display_step
        self.model_path = model_path
        self.test_dataset = test_dataset

    def compute_accuracy(self, sess, X_pl, y_pl, num_correct, dataset, eval_feed):
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

    def get_optimizer(self, lr_pl): 
        optimizer = None 
        if self.optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr_pl)
        elif self.optimizer_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_pl)
        elif self.optimizer_type == 'sgd_mom':
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr_pl, momentum=0.99)
        else:
            raise ValueError("Unknown optimizer.")
        return optimizer 

    def evaluate(self, inputs, outputs, hs):
        tf.reset_default_graph()

        X_pl = tf.placeholder("float", [None] + self.in_dim)
        y_pl = tf.placeholder("float", [None, self.num_classes])
        lr_pl = tf.placeholder("float")
        co.forward({inputs['In'] : X_pl})
        logits = outputs['Out'].val
        train_feed, eval_feed = htf.get_feed_dicts(outputs.values())

        # define loss and optimizer
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_pl))
        optimizer = self.get_optimizer(lr_pl)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = optimizer.minimize(loss)

        # for computing the accuracy of the model
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_pl, 1))
        num_correct = tf.reduce_sum(tf.cast(correct_prediction, "float"))

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            lr = self.learning_rate if 'learning_rate' not in hs else hs['learning_rate'].val

            num_batches = int(self.train_dataset.get_num_examples() / self.batch_size)
            for epoch in range(self.max_num_training_epochs):
                avg_loss = 0.
                for _ in range(num_batches):
                    X_batch, y_batch = self.train_dataset.next_batch(self.batch_size)
                    train_feed.update({X_pl: X_batch, y_pl: y_batch, lr_pl: lr})

                    _, c = sess.run([optimizer, loss], feed_dict=train_feed)
                    avg_loss += c / num_batches

                val_acc = self.compute_accuracy(sess, X_pl, y_pl, num_correct,
                    self.val_dataset, eval_feed)

                # Display logs per epoch step
                if self.log_output_to_terminal and epoch % self.display_step == 0:
                    print("epoch:", '%04d' % (epoch + 1),
                          "loss:", "{:.9f}".format(avg_loss),
                          "validation_accuracy:", "%.5f" % val_acc)

            val_acc = self.compute_accuracy(sess, X_pl, y_pl, num_correct,
                self.val_dataset, eval_feed)

            results = {'validation_accuracy' : val_acc,
                       'num_parameters' : float(htf.get_num_trainable_parameters())
                       }
        return results

if __name__ == "__main__": 
    main() 