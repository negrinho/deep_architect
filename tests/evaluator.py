from __future__ import print_function
import time
from six.moves import xrange
from reduced_toolbox.tb_experiments import SummaryDict
import tests.dataset as ds
import numpy as np
import tensorflow as tf
import darch.core as co
import darch.helpers.tensorflow as tf_helpers
import os
import pickle


class ClassifierEvaluator:
    """Trains and evaluates a classifier on some datasets passed as argument.

    Uses a number of training tricks, namely, early stopping, keeps the model 
    that achieves the best validation performance, reduces the step size 
    after the validation performance fails to increases for some number of 
    epochs.

    """

    def __init__(self, train_dataset, val_dataset, in_d, num_classes, 
            out_folderpath, dataset_type,
            training_epochs_max=200, max_minutes_per_model=180.0, 
            stop_patience=20, rate_patience=7, batch_patience=np.inf, 
            save_patience=2, rate_mult=0.5, batch_mult=2, 
            optimizer_type='adam', sgd_momentum=0.99,
            learning_rate_init=1e-3, learning_rate_min=1e-6, batch_size_init=32, 
            weight_decay_coeff=1e-5,
            display_step=1, output_to_terminal=False, test_dataset=None):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.in_d = list(in_d)
        self.num_classes = num_classes

        self.training_epochs = training_epochs_max
        self.max_minutes_per_model = max_minutes_per_model
        self.display_step = display_step
        self.stop_patience = stop_patience
        self.rate_patience = rate_patience
        self.batch_patience = batch_patience
        self.save_patience = save_patience
        self.rate_mult = rate_mult
        self.batch_mult = batch_mult
        self.learning_rate_init = learning_rate_init
        self.learning_rate_min = learning_rate_min
        self.batch_size_init = batch_size_init
        self.optimizer_type = optimizer_type
        self.output_to_terminal = output_to_terminal
        self.sgd_momentum = sgd_momentum
        self.weight_decay_coeff = weight_decay_coeff
        self.out_folderpath = out_folderpath
        self.test_dataset = test_dataset
        self.dataset_type = dataset_type

    def eval(self, inputs, outputs, hs):
        
        # extracting the values.
        self.optimizer_type = hs['optimizer_type'].val
        self.learning_rate_init = hs['learning_rate_init'].val
        self.rate_mult = hs['rate_mult'].val
        self.rate_patience = hs['rate_patience'].val
        self.stop_patience = hs['stop_patience'].val
        self.learning_rate_min = hs['learning_rate_min'].val
        self.angle_delta = hs['angle_delta'].val
        self.scale_delta = hs['scale_delta'].val
        self.weight_decay_coeff = hs['weight_decay_coeff'].val

        # TODO: come back here to check that it works.
        ds.set_augmentation_fn(self.dataset_type, 
            self.train_dataset, self.val_dataset, self.test_dataset,
            self.angle_delta, self.scale_delta)

        tf.reset_default_graph()
        tf.set_random_seed(0)

        x = tf.placeholder("float", [None] + self.in_d)
        y = tf.placeholder("float", [None, self.num_classes])
        co.forward( {inputs['In'] : x} )
        pred = outputs['Out'].val 
        train_feed, eval_feed = tf_helpers.get_feed_dicts(outputs.values())    

        # below is training code.
        self.model_path = os.path.join(self.out_folderpath, 'model.ckpt')
        learning_rate = tf.placeholder("float")
        learning_rate_val = self.learning_rate_init
        batch_size = self.batch_size_init
        saver = tf.train.Saver()

        # Define loss and optimizer
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

        if self.weight_decay_coeff > 0.0:
            loss = loss + self.weight_decay_coeff * tf.reduce_sum(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            
        # chooses the optimizer. (this can be put in a function).
        if self.optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif self.optimizer_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer_type == 'sgd_mom':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=self.sgd_momentum)
        else:
            raise ValueError("Unknown optimizer.")
        
        # NOTE: in case there is stuff that needs to be implemented.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = optimizer.minimize(loss)

        # For computing the accuracy of the model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        num_correct = tf.reduce_sum(tf.cast(correct_prediction, "float"))
        def compute_accuracy(dataset, ev_feed, ev_batch_size):
            nc = 0
            n_left = dataset.get_num_examples()
            while n_left > 0:
                images, labels = dataset.next_batch(ev_batch_size)
                ev_feed.update({x: images, 
                                y: labels})
                nc += num_correct.eval(ev_feed)
                # update the number of examples left.
                eff_batch_size = labels.shape[0]
                n_left -= eff_batch_size
            
            acc = float(nc) / dataset.get_num_examples()
            return acc 

        # Initializing the variables
        init = tf.global_variables_initializer()

        summary = SummaryDict(abort_if_different_lengths=True)
        # Launch the graph
        with tf.Session(
                #config=tf.ConfigProto(
                #    allow_soft_placement=True
                #)
            ) as sess:
            sess.run(init)

            # for early stopping
            best_vacc = - np.inf
            best_vacc_saved = - np.inf
            stop_counter = self.stop_patience
            rate_counter = self.rate_patience
            batch_counter = self.batch_patience
            save_counter = self.save_patience
            time_start = time.time()

            train_num_examples = self.train_dataset.get_num_examples()
            val_num_examples = self.val_dataset.get_num_examples()

            # Training cycle
            for epoch in xrange(self.training_epochs):
                avg_loss = 0.
                total_batch = int(train_num_examples / batch_size)
                # Loop over all batches
                for i in xrange(total_batch):
                    batch_x, batch_y = self.train_dataset.next_batch(batch_size)
                    #print((batch_x.shape, batch_y.shape))
                    #import ipdb; ipdb.set_trace()
                    # Run optimization op (backprop) and loss op (to get loss value)
                    train_feed.update({x: batch_x, 
                                       y: batch_y, 
                                       learning_rate: learning_rate_val})

                    _, c = sess.run([optimizer, loss], feed_dict=train_feed)
                    # Compute average loss
                    avg_loss += c / total_batch

                    # at the end of the epoch, if spent more time than budget, exit.
                    time_now = time.time()
                    if (time_now - time_start) / 60.0 > self.max_minutes_per_model:
                        break

                # early stopping
                vacc = compute_accuracy(self.val_dataset, eval_feed, batch_size)

                # Display logs per epoch step
                if self.output_to_terminal and epoch % self.display_step == 0:
                    print("Time:", "%7.1f" % (time.time() - time_start),
                          "Epoch:", '%04d' % (epoch+1),
                          "loss=", "{:.9f}".format(avg_loss),
                          "val_acc=", "%.5f" % vacc, 
                          "learn_rate=", '%.3e' % learning_rate_val)
                
                summary.append({
                    'time_seq' : time.time() - time_start, 
                    'val_acc_seq' : vacc,
                    'learn_rate_seq' : learning_rate_val})

                if best_vacc < vacc: 
                    best_vacc = vacc
                    # reinitialize all the counters.
                    stop_counter = self.stop_patience
                    rate_counter = self.rate_patience
                    batch_counter = self.batch_patience
                    save_counter = self.save_patience
                else:
                    stop_counter -= 1
                    rate_counter -= 1
                    batch_counter -= 1
                    if stop_counter == 0:
                        break   

                    if rate_counter == 0:
                        learning_rate_val *= self.rate_mult
                        rate_counter = self.rate_patience

                        if learning_rate_val < self.learning_rate_min:
                            learning_rate_val = self.learning_rate_min

                    if batch_counter == 0:
                        batch_size *= self.batch_mult
                        batch_counter = self.batch_patience

                    if best_vacc_saved < vacc:
                        save_counter -= 1

                        if save_counter == 0:
                            save_path = saver.save(sess, self.model_path)
                            print("Model saved in file: %s" % save_path)

                            save_counter = self.save_patience
                            best_vacc_saved = vacc
                
                time_now = time.time()
                if (time_now - time_start) / 60.0 > self.max_minutes_per_model:
                    break
                
            # if the model saved has better performance than the current model,
            # load it.
            if best_vacc_saved > vacc:
                saver.restore(sess, self.model_path)
                print("Model restored from file: %s" % save_path)

            print("Optimization Finished!")

            # compute the correct predictions to keep them.
            predictions = tf.argmax(pred, 1)
            def compute_predictions(dataset, ev_feed, ev_batch_size):
                nc = 0
                n_left = dataset.get_num_examples()
                pred_lst = []
                while n_left > 0:
                    images, labels = dataset.next_batch(ev_batch_size)
                    ev_feed.update({x: images, 
                                    y: labels})
                    pred_lst.append(predictions.eval(ev_feed))
                    # update the number of examples left.
                    eff_batch_size = labels.shape[0]
                    n_left -= eff_batch_size
                
                return np.concatenate(pred_lst)

            out_d = {}
            out_d.update(summary.get_dict())
            out_d['val_accuracy'] = compute_accuracy(self.val_dataset, 
                eval_feed, batch_size)
            out_d['val_preds'] = compute_predictions(self.val_dataset, 
                eval_feed, batch_size)
            out_d['val_labels'] = ds.onehot_to_idx(self.val_dataset.y)
            print("Validation accuracy: %f" % out_d['val_accuracy'])
            
            if self.test_dataset != None:
                out_d['test_accuracy'] = compute_accuracy(self.test_dataset, 
                    eval_feed, batch_size)
                out_d['test_preds'] = compute_predictions(self.test_dataset, 
                    eval_feed, batch_size)
                out_d['test_labels'] = ds.onehot_to_idx(self.test_dataset.y)
                print("Test accuracy: %f" % out_d['test_accuracy'])

            self.out_d = out_d
            out_path = os.path.join(self.out_folderpath, 'out.pkl')
            with open(out_path, 'wb') as f:
                pickle.dump(out_d, f)
            save_path = saver.save(sess, self.model_path)

        return out_d['val_accuracy']