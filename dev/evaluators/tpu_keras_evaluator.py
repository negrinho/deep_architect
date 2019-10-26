import os
import sys
import gc
import subprocess
import random
import logging

import tensorflow as tf
import numpy as np

from deep_architect.contrib.misc.datasets.cifar10_tf import Cifar10DataSet
import deep_architect.contrib.misc.evaluators.tensorflow.gcloud_utils as gcu
import deep_architect.core as co
import deep_architect.utils as ut
import deep_architect.helpers.tensorflow_eager_support as htfe

logger = logging.getLogger(__name__)


def input_fn(mode, data_dir, batch_size=128, train=True, num_outputs=1):

    data = Cifar10DataSet(data_dir, subset=mode,
                          use_distortion=train).make_batch(batch_size)

    def transform_labels(feat, labels):
        labels = tf.expand_dims(labels, 1)
        return feat, tuple(labels for _ in range(num_outputs))

    return data.map(transform_labels, num_parallel_calls=2)


def get_optimizer(optimizer_type, learning_rate):
    if optimizer_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate)
    elif optimizer_type == 'sgd_mom':
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=.9)
    elif optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                              momentum=.9,
                                              epsilon=1.0)
    return optimizer


class KerasTPUEvaluator:

    def __init__(self,
                 data_dir,
                 tpu_name,
                 max_num_training_epochs=200,
                 stop_patience=20,
                 optimizer_type='adam',
                 batch_size=256,
                 lr_decay_method='constant',
                 init_lr=.001,
                 lr_decay_value=.97,
                 lr_num_epochs_per_decay=2.4,
                 lr_warmup_epochs=3.0,
                 weight_decay=0,
                 display_step=1,
                 log_output_to_terminal=True,
                 base_dir='./scratch',
                 delete_scratch_after_use=False,
                 epochs_between_evals=100,
                 use_tpu=True):
        self.tpu_name = tpu_name
        self.num_examples = Cifar10DataSet.num_examples_per_epoch()
        self.num_examples = 512
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.steps_per_epoch = self.num_examples // self.batch_size
        self.steps_per_val_epoch = Cifar10DataSet.num_examples_per_epoch(
            subset='validation') // self.batch_size
        self.steps_per_val_epoch = 3
        self.steps_per_test_epoch = Cifar10DataSet.num_examples_per_epoch(
            subset='eval') // self.batch_size
        self.max_num_training_epochs = max_num_training_epochs
        self.epochs_between_evals = epochs_between_evals
        self.display_step = display_step
        self.stop_patience = stop_patience
        self.lr_decay_method = lr_decay_method
        self.init_lr = init_lr * 8 if use_tpu else init_lr
        self.lr_decay_value = lr_decay_value
        self.lr_num_epochs_per_decay = lr_num_epochs_per_decay
        self.lr_warmup_epochs = lr_warmup_epochs
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.log_output_to_terminal = log_output_to_terminal
        self.base_dir = base_dir
        self.use_tpu = use_tpu
        self.delete_scratch_after_use = delete_scratch_after_use
        self.num_parameters = -1

    def get_learning_rate(self, step):
        total_steps = int(self.max_num_training_epochs * self.num_examples /
                          self.batch_size)
        if self.lr_decay_method == 'constant':
            lr = self.init_lr
        elif self.lr_decay_method == 'cosine':
            lr = tf.train.cosine_decay(self.init_lr, step, total_steps)
        elif self.lr_decay_method == 'stepwise':
            # divide LR by 10 at 1/2, 2/3, and 5/6 of total epochs
            boundaries = [
                int(0.5 * total_steps),
                int(0.667 * total_steps),
                int(0.833 * total_steps)
            ]
            values = [
                1.0 * self.init_lr, 0.1 * self.init_lr, 0.01 * self.init_lr,
                0.0001 * self.init_lr
            ]
            lr = tf.train.piecewise_constant(step, boundaries, values)
        else:
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

        return lr

    def eval(self, inputs, outputs, save_fn=None, state=None):
        if state is not None and 'model_dir' in state:
            model_dir = state['model_dir']
        else:
            model_dir = gcu.get_empty_bucket_folder(self.base_dir)
            if save_fn:
                save_fn({'model_dir': model_dir})
        logger.info('Using folder %s for evaluation', model_dir)
        latest = tf.train.latest_checkpoint(model_dir)
        init_epoch = 0
        train_dataset = input_fn('train',
                                 self.data_dir,
                                 batch_size=self.batch_size,
                                 train=True,
                                 num_outputs=len(outputs))
        val_dataset = input_fn('validation',
                               self.data_dir,
                               batch_size=self.batch_size,
                               train=False,
                               num_outputs=len(outputs))
        if latest:
            model = tf.keras.models.load_model(latest)
            init_epoch = int(latest.split('.')[-4])
        else:
            input_placeholder = tf.keras.Input(
                train_dataset.output_shapes[0][1:])
            x = input_placeholder
            step = tf.train.get_or_create_global_step()
            if 'in' in inputs:
                co.forward({inputs['in']: x})
                logits = outputs['out'].val
                logits = tf.keras.layers.Lambda(lambda x: x,
                                                name='final_logits')(logits)
                output_tensors = [
                    logits,
                ]
                loss_weights = [1.0]
                losses = [
                    lambda y_true, y_pred: tf.keras.losses.
                    sparse_categorical_crossentropy(
                        y_true, y_pred, from_logits=True),
                ]
                accuracy_metric_name = 'sparse_categorical_accuracy'
            else:
                co.forward({
                    inputs['in0']:
                    x,
                    inputs['in1']:
                    float(self.steps_per_epoch * self.max_num_training_epochs)
                })
                logits = outputs['out1'].val
                aux_logits = outputs['out0'].val
                logits = tf.keras.layers.Lambda(lambda x: x,
                                                name='final_logits')(logits)
                aux_logits = tf.keras.layers.Lambda(lambda x: x,
                                                    name='aux_logits')(
                                                        aux_logits)
                output_tensors = [logits, aux_logits]
                loss_weights = [1.0, .5]
                losses = [
                    lambda y_true, y_pred: tf.keras.losses.
                    sparse_categorical_crossentropy(
                        y_true, y_pred, from_logits=True), lambda y_true,
                    y_pred: tf.keras.losses.sparse_categorical_crossentropy(
                        y_true, y_pred, from_logits=True)
                ]
                accuracy_metric_name = 'final_logits_sparse_categorical_accuracy'
            metrics = {'final_logits': ['sparse_categorical_accuracy']}
            model = tf.keras.Model(inputs=input_placeholder,
                                   outputs=output_tensors)
            if self.use_tpu:
                my_project_name = subprocess.check_output(
                    ['gcloud', 'config', 'get-value', 'project'])
                my_zone = subprocess.check_output(
                    ['gcloud', 'config', 'get-value', 'compute/zone'])
                cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                    self.tpu_name.split(','),
                    zone=my_zone,
                    project=my_project_name)
                strategy = tf.contrib.tpu.TPUDistributionStrategy(
                    cluster_resolver)
                model = tf.contrib.tpu.keras_to_tpu_model(model, strategy)

            learning_rate = self.get_learning_rate(step)
            optimizer = get_optimizer(self.optimizer_type, learning_rate)
            tf.summary.scalar('learning_rate', learning_rate)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=model_dir)
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    model_dir, 'weights.{epoch:02d}.{val_loss:.2f}.hdf5'),
                period=(self.max_num_training_epochs // 3) + 1)
            model.compile(optimizer=optimizer,
                          loss=losses,
                          loss_weights=loss_weights,
                          metrics=metrics)
        num_parameters = int(
            np.sum([
                np.prod(v.get_shape().as_list())
                for v in set(model.trainable_weights)
            ]))
        logger.info('Number of parameters for %s: %d', model_dir,
                    num_parameters)
        # print(model.metrics_names)
        # return model
        timer_manager = ut.TimerManager()
        timer_manager.create_timer('eval')
        # return model

        training_history = model.fit(
            (lambda: train_dataset) if self.use_tpu else train_dataset,
            epochs=self.max_num_training_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=(lambda: val_dataset)
            if self.use_tpu else val_dataset,
            validation_steps=self.steps_per_val_epoch,
            initial_epoch=init_epoch,
            callbacks=[tensorboard_callback, checkpoint_callback],
        )
        test_dataset = input_fn('eval',
                                self.data_dir,
                                batch_size=self.batch_size,
                                train=False,
                                num_outputs=len(outputs))
        test_results = model.evaluate(
            (lambda: test_dataset) if self.use_tpu else test_dataset,
            steps=5)  #self.steps_per_test_epoch)
        test_results = {
            'test_' + metric_name: test_results[ix]
            for (ix, metric_name) in enumerate(model.metrics_names)
        }

        results = {
            'validation_accuracy':
            training_history.history['val_' + accuracy_metric_name][-1],
            'validation_loss':
            training_history.history['val_loss'][-1],
            'num_parameters':
            num_parameters,
            'num_training_epochs':
            training_history.epoch,
        }
        results.update(test_results)
        results.update(training_history.history)
        results['test_accuracy'] = test_results['test_' + accuracy_metric_name]

        results['training_time_in_hours'] = timer_manager.get_time_since_event(
            'eval', 'start', units='hours')
        return results
