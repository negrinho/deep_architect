
from __future__ import print_function
from __future__ import division

import gc
import os
from builtins import object
from past.utils import old_div
import tensorflow as tf
tfe = tf.contrib.eager
import numpy as np
import darch.core as co
import darch.search_logging as sl
import darch.contrib.useful.gpu_utils as gpu_utils
from dev.architecture_search_benchmarks.helpers.tfeager import setTraining
from six.moves import range
import time

class ENASEagerEvaluator(object):
    """Trains and evaluates a classifier on some datasets passed as argument.
    Uses a number of training tricks, namely, early stopping, keeps the model
    that achieves the best validation performance, reduces the step size
    after the validation performance fails to increases for some number of
    epochs.
    """

    def __init__(self, train_dataset, val_dataset, num_classes, weight_sharer, 
            optimizer_type='adam', batch_size=128,
            learning_rate_init=1e-3, display_step=50, log_output_to_terminal=True,
            test_dataset=None, max_controller_steps=50):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.in_dim = list(train_dataset.next_batch(1)[0].shape[1:])

        self.display_step = display_step
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.log_output_to_terminal = log_output_to_terminal
        self.test_dataset = test_dataset
        self.num_batches = int(old_div(self.train_dataset.get_num_examples(), self.batch_size))
        self.batch_counter = 0
        self.epoch = 0

        self.controller_step = 0
        self.child_step = 0
        self.controller_mode = False
        self.max_controller_steps = max_controller_steps
        
        self.weight_sharer = weight_sharer

        if self.optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_init)
        elif self.optimizer_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_init)
        elif self.optimizer_type == 'sgd_mom':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_init, momentum=0.99)
        else:
            raise ValueError("Unknown optimizer.")

    def save_state(self, folderpath):
        weight_sharer_file = os.path.join(folderpath, "weight_sharer")
        self.weight_sharer.save(weight_sharer_file)
    
    def load_state(self, folderpath):
        weight_sharer_file = os.path.join(folderpath, "weight_sharer")
        self.weight_sharer.load(weight_sharer_file)

    def _compute_accuracy(self, inputs, outputs, dataset):
        nc = 0
        num_left = dataset.get_num_examples()
        t0 = time.time()
        setTraining(list(outputs.values()), False)
        t00 = time.time()
        loss = 0
        time1 = 0.0
        time2 = 0.0
        time3 = 0.0
        time4 = 0.0
        time5 = 0.0
        time6 = 0.0
        while num_left > 0:
            t1 = time.time()
            X_batch, y_batch = dataset.next_batch(self.batch_size)
            t2 = time.time()
            X = tf.convert_to_tensor(X_batch, np.float32)#.gpu()
            t3 = time.time()
            y = tf.convert_to_tensor(y_batch, np.float32)#.gpu()

            t4 = time.time()
            co.forward({inputs['In']: X})
            t5 = time.time()
            logits = outputs['Out'].val
            t4 = time.time()
            
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            t5 = time.time()
            num_correct = tf.reduce_sum(tf.cast(correct_prediction, "float"))
            t6 = time.time()
            loss += tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
            t7 = time.time()
            nc += num_correct
            time1 += t2 - t1
            time2 += t3 - t2
            time3 += t4 - t3
            time4 += t5 - t4
            time5 += t6 - t5
            time6 += t7 - t6
            
            # update the number of examples left.
            eff_batch_size = y_batch.shape[0]
            num_left -= eff_batch_size
        t8 = time.time()
        acc = old_div(float(nc), dataset.get_num_examples())
        loss = old_div(float(loss), dataset.get_num_examples())
        t9 = time.time()
        print('time0 = ' + str(t00 - t0))
        print('time1 = ' + str(time1))
        print('time2 = ' + str(time2))
        print('time3 = ' + str(time3))
        print('time4 = ' + str(time4))
        print('time5 = ' + str(time5))
        print('time6 = ' + str(time6))
        print('time7 = ' + str(t9 - t8))

        return acc, loss

    def _compute_loss(self, inputs, outputs, X, y, loss_metric):
        X = tf.constant(X).gpu()
        y = tf.constant(y).gpu()
        co.forward({inputs['In']: X})
        logits = outputs['Out'].val
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        loss_metric(loss)
        return loss
        

    def eval(self, inputs, outputs, hs):
        results = {}
        if self.controller_mode:
            # Compute accuracy of model
            t1 = time.time()
            val_acc, loss = self._compute_accuracy(inputs, outputs, self.val_dataset)
            t2 = time.time()
            results['validation_accuracy'] = val_acc

            # Log validation info
            self.controller_step += 1
            if self.log_output_to_terminal:# and self.controller_step % self.display_step == 0:
                log_string = ""
                log_string += "ctrl_step={:<6d}".format(self.controller_step)
                log_string += " loss={:<7.3f}".format(loss)
                log_string += " acc={:<6.4f}".format(val_acc)
                log_string += " time={:<6.4f}".format(t2 - t1)
                print(log_string)

            # If controller phase finished, update epoch and switch back to 
            # updating child params
            if self.controller_step % self.max_controller_steps == 0:
                self.controller_mode = False
                self.epoch += 1
                results['epoch'] = self.epoch
                print('Starting Image model mode')

        else:
            # Update child model parameters
            X_batch, y_batch = self.train_dataset.next_batch(self.batch_size)
            setTraining(list(outputs.values()), True)
            with tf.device('/gpu:0'):
                loss_metric = tfe.metrics.Mean('loss')
                self.optimizer.minimize(lambda: self._compute_loss(inputs, outputs, X_batch, y_batch, loss_metric))

            # Log batch info
            self.child_step += 1
            if self.log_output_to_terminal and self.child_step % self.display_step == 0:
                log_string = ""
                log_string += "epoch={:<6d}".format(self.epoch)
                log_string += " ch_step={:<6d}".format(self.child_step)
                log_string += " loss={:<8.6f}".format(loss_metric.result())
                print(log_string)
            epoch_end = self.train_dataset.iter_i == 0 or self.train_dataset.iter_i > 25 * 128

            # If epoch completed, switch to updating controller
            results['validation_accuracy'] = -1
            if epoch_end:
                self.controller_mode = True
                print('Starting Controller Mode')
        gc.collect()
        return results
