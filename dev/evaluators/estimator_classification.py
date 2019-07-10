# TODO move to dev
from __future__ import print_function

import os
import shutil
import gc
import tensorflow as tf

import numpy as np
import deep_architect.core as co
import deep_architect.helpers.tensorflow as htf
import deep_architect.utils as ut
import deep_architect.contrib.misc.gpu_utils as gpu_utils
import deep_architect.contrib.misc.datasets.augmentation as aug
import deep_architect.helpers.tfeager_support as htfe
from six.moves import range
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3


def get_feature_columns():
    feature_columns = {
        'images':
        tf.feature_column.numeric_column(
            'images', (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)),
    }
    return feature_columns


def setRecompile(output_lst, recompile):

    def fn(mx):
        mx._is_compiled = not recompile

    co.traverse_backward(output_lst, fn)


def input_fn(features, labels, batch_size=128, train=True):
    features = {'images': features}
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if train:

        def augmentation(image, label):
            image = tf.random_crop(
                tf.image.resize_images(image['images'], (40, 40)), (32, 32, 3))
            return {'images': tf.image.random_flip_left_right(image)}, label

        dataset = dataset.shuffle(10 * batch_size)
        dataset = dataset.map(augmentation, num_parallel_calls=8)
    dataset = dataset.batch(batch_size).prefetch(2 * batch_size)
    return dataset


def delete_files_in_folder(folder):
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)


class AdvanceClassifierEvaluator:

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 num_classes,
                 max_num_training_epochs=200,
                 stop_patience=20,
                 optimizer_type='sgd_mom',
                 batch_size=128,
                 whiten=False,
                 init_lr=.015,
                 lr_decay_value=.97,
                 lr_num_epochs_per_decay=2.4,
                 lr_warmup_epochs=3.0,
                 weight_decay=.001,
                 display_step=1,
                 log_output_to_terminal=True,
                 test_dataset=None,
                 base_dir='scratch'):

        self.X_train = train_dataset.X
        self.X_val = val_dataset.X
        self.X_test = test_dataset.X if test_dataset else None
        self.y_train = train_dataset.y
        self.y_val = val_dataset.y
        self.y_test = test_dataset.y if test_dataset else None
        if whiten:
            self.X_train = aug.per_image_whiten(self.X_train)
            self.X_val = aug.per_image_whiten(self.X_val)
            self.X_test = aug.per_image_whiten(
                self.X_test) if self.X_test is None else None

        self.in_dim = list(train_dataset.next_batch(1)[0].shape[1:])
        self.num_examples = self.X_train.shape[0]
        self.batch_size = batch_size
        self.steps_per_epoch = (self.num_examples + self.batch_size -
                                1) / self.batch_size
        self.max_num_training_epochs = max_num_training_epochs
        self.display_step = display_step
        self.stop_patience = stop_patience
        self.init_lr = init_lr
        self.lr_decay_value = lr_decay_value
        self.lr_num_epochs_per_decay = lr_num_epochs_per_decay
        self.lr_warmup_epochs = lr_warmup_epochs
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.log_output_to_terminal = log_output_to_terminal
        self.base_dir = base_dir
        self.num_archs = 0
        ut.create_folder(base_dir, abort_if_exists=False)
        delete_files_in_folder(base_dir)

    def get_learning_rate(self, step):
        lr = tf.train.exponential_decay(self.init_lr,
                                        step,
                                        self.steps_per_epoch *
                                        self.lr_num_epochs_per_decay,
                                        self.lr_decay_value,
                                        staircase=True)
        warmup_steps = int(self.lr_warmup_epochs * self.steps_per_epoch)
        warmup_lr = (self.init_lr * tf.cast(step, tf.float32) /
                     tf.cast(warmup_steps, tf.float32))
        lr = tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: lr)
        lr = tf.maximum(lr, 0.0001 * self.init_lr)
        tf.summary.scalar('learning_rate', lr)

        return lr

    def eval(self, inputs, outputs):
        tf.reset_default_graph()

        model_dir = ut.join_paths([self.base_dir, 'eval' + str(self.num_archs)])
        ut.create_folder(model_dir, abort_if_exists=False)

        def model_fn(features, labels, mode, params):
            feature_columns = list(get_feature_columns().values())

            images = tf.feature_column.input_layer(
                features=features, feature_columns=feature_columns)

            images = tf.reshape(images,
                                shape=(-1, IMAGE_HEIGHT, IMAGE_WIDTH,
                                       IMAGE_DEPTH))
            setRecompile(outputs.values(), True)
            gc.collect()
            htfe.setTraining(outputs.values(),
                             mode == tf.estimator.ModeKeys.TRAIN)
            co.forward({inputs['In']: images})
            logits = outputs['Out'].val

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            predicted_classes = tf.argmax(logits, 1)
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'class_ids': predicted_classes[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits,
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            # define loss and optimizer
            train_vars = tf.trainable_variables()
            with tf.variable_scope('l2'):
                l2_loss = tf.add_n([
                    tf.nn.l2_loss(v) for v in train_vars if 'kernel' in v.name
                ]) * self.weight_decay
            unreg_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=labels))
            loss = unreg_loss + l2_loss
            # Compute evaluation metrics.
            accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                           predictions=predicted_classes,
                                           name='acc_op')
            metrics = {'accuracy': accuracy}
            if mode == tf.estimator.ModeKeys.EVAL:
                loss = tf.Print(loss, [
                    accuracy, l2_loss, unreg_loss, loss,
                    tf.argmax(labels, 1), predicted_classes
                ],
                                summarize=10)
                return tf.estimator.EstimatorSpec(mode,
                                                  loss=loss,
                                                  eval_metric_ops=metrics)

            # Create training op.
            assert mode == tf.estimator.ModeKeys.TRAIN
            step = tf.train.get_or_create_global_step()
            learning_rate = self.get_learning_rate(step)
            optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                                  .9,
                                                  momentum=.9,
                                                  epsilon=1.0)
            loss = tf.Print(loss, [
                accuracy, l2_loss, unreg_loss, loss, learning_rate,
                tf.argmax(labels, 1), predicted_classes
            ],
                            summarize=10)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode,
                                              loss=loss,
                                              train_op=train_op)

        # NUM_GPUS = 2
        # strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
        # config = tf.estimator.RunConfig(train_distribute=strategy)
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                            session_config=config)
        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                           config=run_config,
                                           params={})
        seqs = ut.SequenceTracker(abort_if_different_lengths=True)

        best_val_acc = -np.inf
        stop_counter = self.stop_patience
        timer_manager = ut.TimerManager()
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

        for epoch in range(self.max_num_training_epochs):
            train_fn = lambda: input_fn(self.X_train, self.y_train, train=True)
            val_fn = lambda: input_fn(self.X_val, self.y_val, train=False)
            print('\n\nTraining')
            estimator.train(input_fn=train_fn)
            print('\n\nEvaluating')
            eval_results = estimator.evaluate(input_fn=val_fn)

            # early stopping
            val_acc = eval_results['accuracy']

            # Display logs per epoch step
            if self.log_output_to_terminal and epoch % self.display_step == 0:
                print(
                    "time:", "%7.1f" %
                    timer_manager.get_time_since_event('eval', 'start'),
                    "epoch:", '%04d' % (epoch + 1), "validation loss:",
                    "{:.9f}".format(eval_results['loss']),
                    "validation_accuracy:", "%.5f" % val_acc)

            d = {
                'validation_accuracy':
                val_acc,
                'validation_loss':
                eval_results['loss'],
                'epoch_number':
                epoch + 1,
                'time_in_minutes':
                timer_manager.get_time_since_event('eval',
                                                   'start',
                                                   units='minutes'),
            }
            seqs.append(d)

            # update the patience counters.
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                # reinitialize all the counters.
                stop_counter = self.stop_patience
            else:
                stop_counter -= 1
                if stop_counter == 0:
                    break

        print("Optimization Finished!")

        timer_manager.tick_timer('eval')
        eval_results = estimator.evaluate(
            input_fn=lambda: input_fn(self.X_val, self.y_val))

        val_acc = eval_results['accuracy']
        t_infer = (
            timer_manager.get_time_since_last_tick('eval', 'miliseconds') /
            self.X_val.shape[0])

        print("Validation accuracy: %f" % val_acc)
        seqs_dict = seqs.get_dict()
        results = {
            'validation_accuracy': val_acc,
            'num_parameters': float(htf.get_num_trainable_parameters()),
            'inference_time_per_example_in_miliseconds': t_infer,
            'num_training_epochs': seqs_dict['epoch_number'],
            'sequences': seqs_dict
        }
        if 'gpu_utilization_in_percent' in seqs_dict:
            results['average_gpu_utilization_in_percent'] = np.mean(
                seqs_dict['gpu_utilization_in_percent'])
            results['average_gpu_memory_utilization_in_gigabytes'] = np.mean(
                seqs_dict['gpu_memory_utilization_in_gigabytes'])

        if self.X_test != None and self.y_test != None:
            test_results = estimator.evaluate(
                input_fn=lambda: input_fn(self.X_test, self.y_test))
            test_acc = test_results['accuracy']
            print("Test accuracy: %f" % test_acc)
            results['test_accuracy'] = test_acc

        results['training_time_in_hours'] = timer_manager.get_time_since_event(
            'eval', 'start', units='hours')
        self.num_archs += 1
        return results

    def save_state(self, folder):
        pass

    def load_state(self, folder):
        pass