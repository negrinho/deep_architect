
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import darch.core as co
import darch.helpers.tensorflow as htf
import darch.search_logging as sl
import darch.contrib.gpu_utils as gpu_utils

class SimpleClassifierEvaluator:
    """Trains and evaluates a classifier on some datasets passed as argument.
    Uses a number of training tricks, namely, early stopping, keeps the model
    that achieves the best validation performance, reduces the step size
    after the validation performance fails to increases for some number of
    epochs.
    """

    def __init__(self, train_dataset, val_dataset, num_classes, model_path,
            max_num_training_epochs=200, max_eval_time_in_minutes=180.0,
            stop_patience=20, save_patience=2,
            optimizer_type='adam', batch_size=256,
            learning_rate_patience=7, learning_rate_init=1e-3,
            learning_rate_min=1e-6, learning_rate_mult=0.1,
            display_step=1, log_output_to_terminal=True, test_dataset=None):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.in_dim = list(train_dataset.next_batch(1)[0].shape[1:])

        self.max_num_training_epochs = max_num_training_epochs
        self.max_eval_time_in_minutes = max_eval_time_in_minutes
        self.display_step = display_step
        self.stop_patience = stop_patience
        self.save_patience = save_patience
        self.learning_rate_patience = learning_rate_patience
        self.learning_rate_mult = learning_rate_mult
        self.learning_rate_init = learning_rate_init
        self.learning_rate_min = learning_rate_min
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.log_output_to_terminal = log_output_to_terminal
        self.model_path = model_path
        self.test_dataset = test_dataset

    def _compute_accuracy(self, sess, X_pl, y_pl, num_correct, dataset, eval_feed):
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

    def eval(self, inputs, outputs, hs):
        tf.reset_default_graph()

        X_pl = tf.placeholder("float", [None] + self.in_dim)
        y_pl = tf.placeholder("float", [None, self.num_classes])
        lr_pl = tf.placeholder("float")
        co.forward({inputs['In'] : X_pl})
        logits = outputs['Out'].val
        train_feed, eval_feed = htf.get_feed_dicts(outputs.values())
        saver = tf.train.Saver()

        # define loss and optimizer
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_pl))
        # chooses the optimizer. (this can be put in a function).
        if self.optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr_pl)
        elif self.optimizer_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_pl)
        elif self.optimizer_type == 'sgd_mom':
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr_pl, momentum=0.99)
        else:
            raise ValueError("Unknown optimizer.")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = optimizer.minimize(loss)

        # for computing the accuracy of the model
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_pl, 1))
        num_correct = tf.reduce_sum(tf.cast(correct_prediction, "float"))

        init = tf.global_variables_initializer()
        seqs = sl.SequenceTracker(abort_if_different_lengths=True)
        with tf.Session() as sess:
            sess.run(init)

            learning_rate_init = self.learning_rate_init if 'learning_rate_init' not in hs else hs['learning_rate_init'].val
            learning_rate_mult = self.learning_rate_mult if 'learning_rate_mult' not in hs else hs['learning_rate_mult'].val
            learning_rate_min = self.learning_rate_min if 'learning_rate_min' not in hs else hs['learning_rate_min'].val
            stop_patience = self.stop_patience if 'stop_patience' not in hs else hs['stop_patience'].val
            learning_rate_patience = self.learning_rate_patience if 'learning_rate_patience' not in hs else hs['learning_rate_patience'].val
            save_patience = self.save_patience if 'save_patience' not in hs else hs['save_patience'].val

            best_val_acc = - np.inf
            best_val_acc_saved = - np.inf
            stop_counter = stop_patience
            rate_counter = learning_rate_patience
            save_counter = save_patience
            timer_manager = sl.TimerManager()
            timer_manager.create_timer('eval')

            # getting the gpu_id based on the environment.
            if gpu_utils.is_environment_variable_defined('CUDA_VISIBLE_DEVICES'):
                s = gpu_utils.get_environment_variable('CUDA_VISIBLE_DEVICES')
                s_lst = s.split(',')
                if len(s_lst) == 1 and len(s_lst[0]) > 0:
                    gpu_id = int(s_lst[0])
                else:
                    gpu_id = None
            else:
                gpus = gpu_utils.get_gpu_information()
                if len(gpus) == 1:
                    gpu_id = 0
                else:
                    gpu_id = None

            lr = learning_rate_init
            num_batches = int(self.train_dataset.get_num_examples() / self.batch_size)
            for epoch in xrange(self.max_num_training_epochs):
                avg_loss = 0.
                for _ in xrange(num_batches):
                    X_batch, y_batch = self.train_dataset.next_batch(self.batch_size)
                    train_feed.update({X_pl: X_batch, y_pl: y_batch, lr_pl: lr})

                    _, c = sess.run([optimizer, loss], feed_dict=train_feed)
                    avg_loss += c / num_batches

                    # if spent more time than budget, exit.
                    if (timer_manager.get_time_since_event(
                        'eval', 'start', 'minutes') > self.max_eval_time_in_minutes):
                        break

                # early stopping
                val_acc = self._compute_accuracy(sess, X_pl, y_pl, num_correct,
                    self.val_dataset, eval_feed)

                # Display logs per epoch step
                if self.log_output_to_terminal and epoch % self.display_step == 0:
                    print("time:", "%7.1f" % timer_manager.get_time_since_event('eval', 'start'),
                          "epoch:", '%04d' % (epoch + 1),
                          "loss:", "{:.9f}".format(avg_loss),
                          "validation_accuracy:", "%.5f" % val_acc,
                          "learning_rate:", '%.3e' % lr)

                d = {
                    'validation_accuracy' : val_acc,
                    'training_loss' : avg_loss,
                    'epoch_number' : epoch + 1,
                    'learning_rate' : lr,
                    'time_in_minutes' : timer_manager.get_time_since_event('eval', 'start', units='minutes'),
                }
                # adding information about gpu utilization if available.
                if gpu_id is not None:
                    gpus = gpu_utils.get_gpu_information()
                    d.update({
                        'gpu_utilization_in_percent' : gpus[gpu_id]['gpu_utilization_in_percent'],
                        'gpu_memory_utilization_in_gigabytes' : gpus[gpu_id]['gpu_memory_utilization_in_gigabytes']
                    })
                seqs.append(d)

                # update the patience counters.
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    # reinitialize all the counters.
                    stop_counter = stop_patience
                    rate_counter = learning_rate_patience
                    save_counter = save_patience
                else:
                    stop_counter -= 1
                    rate_counter -= 1
                    if stop_counter == 0:
                        break

                    if rate_counter == 0:
                        lr = max(lr * learning_rate_mult, learning_rate_min)
                        rate_counter = learning_rate_patience

                    if best_val_acc_saved < val_acc:
                        save_counter -= 1
                        if save_counter == 0:
                            save_path = saver.save(sess, self.model_path)
                            print("Model saved in file: %s" % save_path)

                            save_counter = save_patience
                            best_val_acc_saved = val_acc

                # if spent more time than budget, exit.
                if (timer_manager.get_time_since_event(
                    'eval', 'start', 'minutes') > self.max_eval_time_in_minutes):
                    break

            # if the model saved has better performance than the current model,
            # load it.
            if best_val_acc_saved > val_acc:
                saver.restore(sess, self.model_path)
                print("Model restored from file: %s" % save_path)

            print("Optimization Finished!")

            timer_manager.tick_timer('eval')
            val_acc = self._compute_accuracy(sess, X_pl, y_pl, num_correct,
                self.val_dataset, eval_feed)
            t_infer = (timer_manager.get_time_since_last_tick(
                'eval', 'miliseconds') / self.val_dataset.get_num_examples())

            print("Validation accuracy: %f" % val_acc)
            seqs_dict = seqs.get_dict()
            results = {'validation_accuracy' : val_acc,
                       'num_parameters' : htf.get_num_trainable_parameters(),
                       'inference_time_per_example_in_miliseconds' : t_infer,
                       'num_training_epochs' : seqs_dict['epoch_number'],
                       'sequences' : seqs_dict
                       }
            if 'gpu_utilization_in_percent' in seqs_dict:
                seqs_dict['average_gpu_utilization_in_percent'] = np.mean(
                    seqs_dict['gpu_utilization_in_percent']),
                seqs_dict['average_gpu_memory_utilization_in_gigabytes'] = np.mean(
                    seqs_dict['gpu_memory_utilization_in_gigabytes']),

            if self.test_dataset != None:
                test_acc = self._compute_accuracy(sess, X_pl, y_pl, num_correct,
                    self.test_dataset, eval_feed)
                print("Test accuracy: %f" % test_acc)
                results['test_accuracy'] = test_acc

        results['training_time_in_hours'] = timer_manager.get_time_since_event(
            'eval', 'start', units='hours')
        return results