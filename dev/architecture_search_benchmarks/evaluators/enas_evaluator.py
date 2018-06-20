
from __future__ import print_function

import tensorflow as tf
import numpy as np
import darch.core as co
import darch.helpers.tensorflow as htf
import darch.search_logging as sl
import darch.contrib.useful.gpu_utils as gpu_utils
from six.moves import range

class ENASEvaluator:
    """Trains and evaluates a classifier on some datasets passed as argument.
    Uses a number of training tricks, namely, early stopping, keeps the model
    that achieves the best validation performance, reduces the step size
    after the validation performance fails to increases for some number of
    epochs.
    """

    def __init__(self, train_dataset, val_dataset, num_classes, weight_sharer, 
            model_path=None, max_num_training_epochs=200, 
            max_eval_time_in_minutes=180.0, stop_patience=20, save_patience=2,
            optimizer_type='adam', batch_size=128,
            learning_rate_patience=7, learning_rate_init=1e-3,
            learning_rate_min=1e-6, learning_rate_mult=0.1,
            display_step=1, log_output_to_terminal=True, test_dataset=None,
            max_controller_steps=2000):

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
        self.num_batches = int(self.train_dataset.get_num_examples() / self.batch_size)
        self.batch_counter = 0
        self.epoch = 0

        self.step = 0
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
        X_pl = tf.placeholder("float", [None] + self.in_dim, name='X_data')
        y_pl = tf.placeholder("float", [None, self.num_classes], name='y_data')
    
        with tf.variable_scope('image_model'):
            co.forward({inputs['In'] : X_pl})
        logits = outputs['Out'].val
        train_feed, eval_feed = htf.get_feed_dicts(outputs.values())

        # define loss and optimizer
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_pl))
        # chooses the optimizer. (this can be put in a function).
        
        tf_variables = [var
                        for var in tf.trainable_variables() if var.name.startswith('image_model')]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = self.optimizer.minimize(loss, var_list=tf_variables)

        # for computing the accuracy of the model
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_pl, 1))
        num_correct = tf.reduce_sum(tf.cast(correct_prediction, "float"))

        init = tf.global_variables_initializer()
        
        results = {}
        # Setting the session to allow growth, so it doesn't allocate all GPU memory.
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        with tf.Session(config=config) as sess:
            sess.run(init)

            if self.controller_mode:
                print('Controller Mode')
                val_acc = self._compute_accuracy(sess, X_pl, y_pl, num_correct,
                    self.val_dataset, eval_feed)

                results['validation_accuracy'] = val_acc
                self.step += 1
                if self.step % self.max_controller_steps == 0:
                    self.controller_mode = False

            else:
                print('Image model mode')
                X_batch, y_batch = self.train_dataset.next_batch(self.batch_size)
                epoch_end = self.train_dataset.iter_i == 0 #or self.train_dataset.iter_i % 1 == 0
                train_feed.update({X_pl: X_batch, y_pl: y_batch})
                _ = sess.run([optimizer], feed_dict=train_feed)
                results['validation_accuracy'] = -1
                if epoch_end:
                    self.controller_mode = True
            self.weight_sharer.update(sess)

        return results