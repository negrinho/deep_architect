from __future__ import print_function

import gc
import logging

import tensorflow as tf
import numpy as np

from deep_architect.contrib.misc.datasets.cifar10_tf import Cifar10DataSet
import deep_architect.contrib.misc.evaluators.tensorflow.gcloud_utils as gcu
import deep_architect.core as co
import deep_architect.utils as ut
import deep_architect.helpers.tensorflow_eager_support as htfe

logger = logging.getLogger(__name__)

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3


def set_recompile(output_lst, recompile):

    def fn(mx):
        mx._is_compiled = not recompile

    co.traverse_backward(output_lst, fn)
    logger.debug('set_recompile')


def input_fn(mode, data_dir, batch_size=128, train=True):
    data = Cifar10DataSet(data_dir, subset=mode,
                          use_distortion=train).make_batch(batch_size)
    return data


def record_summaries(metric_dict, step):
    for key, value in metric_dict.items():
        tf.contrib.summary.scalar(name=key, tensor=value, step=step)


def construct_host_fn(metric_dict, model_dir, prefix='', max_queue_size=10):
    metric_names = list(metric_dict.keys())

    def host_fn(gs, *args):
        step = gs[0]
        with tf.contrib.summary.create_file_writer(
                logdir=model_dir, max_queue=max_queue_size).as_default():
            with tf.contrib.summary.always_record_summaries():
                for i, metric in enumerate(metric_names):
                    tf.contrib.summary.scalar(prefix + metric,
                                              args[i][0],
                                              step=step)
                return tf.contrib.summary.all_summary_ops()

    gs_t = tf.reshape(tf.train.get_or_create_global_step(), [1])
    other_tensors = [tf.reshape(metric_dict[key], [1]) for key in metric_names]
    return host_fn, [gs_t] + other_tensors


def get_optimizer(optimizer_type, learning_rate):
    if optimizer_type == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer_type == 'sgd_mom':
        return tf.train.MomentumOptimizer(learning_rate, momentum=.9)
    elif optimizer_type == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate,
                                         momentum=.9,
                                         epsilon=1.0)
    else:
        raise ValueError('Optimizer type not recognized: %s' % optimizer_type)


class TPUEstimatorEvaluator:

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
                 weight_decay=0.0,
                 display_step=1,
                 log_output_to_terminal=True,
                 base_dir='./scratch',
                 delete_scratch_after_use=False,
                 epochs_between_evals=100,
                 use_tpu=True):
        self.tpu_name = tpu_name
        self.num_examples = Cifar10DataSet.num_examples_per_epoch()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.steps_per_epoch = self.num_examples // self.batch_size
        self.steps_per_val_epoch = Cifar10DataSet.num_examples_per_epoch(
            subset='validation') / self.batch_size
        self.steps_per_test_epoch = Cifar10DataSet.num_examples_per_epoch(
            subset='eval') / self.batch_size
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
        tf.reset_default_graph()
        self.num_parameters = -1
        logger.debug('In Evaluator')
        if state is not None and 'model_dir' in state:
            model_dir = state['model_dir']
        else:
            model_dir = gcu.get_empty_bucket_folder(self.base_dir)
            if save_fn:
                save_fn({'model_dir': model_dir})
        logger.info('Using folder %s for evaluation', model_dir)

        def metric_fn(labels, predictions):
            return {'accuracy': tf.metrics.accuracy(labels, predictions)}

        def model_fn(features, labels, mode, params):
            set_recompile(outputs.values(), True)
            gc.collect()
            htfe.set_is_training(outputs.values(),
                                 mode == tf.estimator.ModeKeys.TRAIN)
            step = tf.train.get_or_create_global_step()
            if 'In' in inputs:
                co.forward({inputs['In']: features})
                logits = outputs['Out'].val
            else:
                co.forward({
                    inputs['In0']:
                    features,
                    inputs['In1']:
                    float(self.steps_per_epoch * self.max_num_training_epochs)
                })
                logits = outputs['Out1'].val
                aux_logits = outputs['Out0'].val

            predicted_classes = tf.argmax(logits, 1, output_type=tf.int32)
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
            onehot_labels = tf.one_hot(labels, 10)
            unreg_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=logits,
                reduction=tf.losses.Reduction.MEAN)
            aux_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=aux_logits,
                weights=.5,
                reduction=tf.losses.Reduction.MEAN) if 'Out1' in outputs else 0
            loss = unreg_loss + l2_loss + aux_loss
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.contrib.tpu.TPUEstimatorSpec(
                    mode,
                    loss=loss,
                    eval_metrics=(metric_fn, [labels, predicted_classes]))
            # Create training op.
            assert mode == tf.estimator.ModeKeys.TRAIN
            if self.num_parameters == -1:
                self.num_parameters = np.sum([
                    np.prod(v.get_shape().as_list())
                    for v in tf.trainable_variables()
                ])
            accuracy = metric_fn(labels, predicted_classes)['accuracy']
            tf.identity(accuracy[1], name='train_accuracy')
            learning_rate = self.get_learning_rate(step)
            metric_dict = {
                'batch_loss':
                loss,
                'learning_rate':
                learning_rate,
                'batch_accuracy':
                tf.reduce_mean(
                    tf.cast(tf.equal(predicted_classes, labels), tf.float32))
            }

            host_fn = None
            optimizer = get_optimizer(self.optimizer_type, learning_rate)
            if self.use_tpu:
                host_fn = construct_host_fn(metric_dict,
                                            model_dir,
                                            prefix='training/',
                                            max_queue_size=self.steps_per_epoch)
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
            else:
                record_summaries(metric_dict, step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                gvs = optimizer.compute_gradients(loss)
                gvs = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad),
                                 grad), val) for grad, val in gvs]
                train_op = optimizer.apply_gradients(
                    gvs, global_step=tf.train.get_or_create_global_step())

            return tf.contrib.tpu.TPUEstimatorSpec(mode,
                                                   loss=loss,
                                                   train_op=train_op,
                                                   host_call=host_fn)

        my_project_name = gcu.get_gcloud_project()
        my_zone = gcu.get_gcloud_zone()
        cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            self.tpu_name.split(','), zone=my_zone, project=my_project_name)
        run_config = tf.contrib.tpu.RunConfig(
            cluster=cluster_resolver,
            model_dir=model_dir,
            save_checkpoints_steps=self.max_num_training_epochs *
            self.steps_per_epoch,
            keep_checkpoint_max=1,
            log_step_count_steps=self.steps_per_epoch,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.steps_per_epoch, num_shards=8),
        )
        results = {}
        try:
            estimator = tf.contrib.tpu.TPUEstimator(
                model_fn=model_fn,
                config=run_config,
                use_tpu=self.use_tpu,
                train_batch_size=self.batch_size,
                eval_batch_size=self.batch_size,
                predict_batch_size=self.batch_size,
                params={})
            timer_manager = ut.TimerManager()
            timer_manager.create_timer('eval')

            train_fn = lambda params: input_fn('train',
                                               self.data_dir,
                                               batch_size=params['batch_size'],
                                               train=True)
            val_fn = lambda params: input_fn('validation',
                                             self.data_dir,
                                             batch_size=params['batch_size'],
                                             train=False)
            try:
                estimator.train(
                    input_fn=train_fn,
                    max_steps=2  #self.steps_per_epoch *
                    # self.max_num_training_epochs
                )
            except (tf.train.NanLossDuringTrainingError,
                    tf.errors.InvalidArgumentError):
                logger.warning(
                    'Architecture in %s received nan loss in training',
                    model_dir)

            logger.debug("Optimization Finished!")

            val_fn = lambda params: input_fn('validation',
                                             self.data_dir,
                                             batch_size=params['batch_size'],
                                             train=False)

            timer_manager.tick_timer('eval')
            eval_results = estimator.evaluate(
                input_fn=val_fn, steps=2)  #self.steps_per_val_epoch)
            t_infer = (
                timer_manager.get_time_since_last_tick('eval', 'miliseconds') /
                Cifar10DataSet.num_examples_per_epoch('validation'))

            val_acc = float(eval_results['accuracy'])
            logger.debug("Validation accuracy: %f", val_acc)

            results = {
                'validation_accuracy': val_acc,
                'num_parameters': self.num_parameters,
                'inference_time_per_example_in_miliseconds': t_infer,
                'epoch': int(eval_results['global_step']) / self.steps_per_epoch
            }
            test_fn = lambda params: input_fn('eval',
                                              self.data_dir,
                                              batch_size=params['batch_size'],
                                              train=False)

            test_results = estimator.evaluate(
                input_fn=test_fn,
                steps=2,  #self.steps_per_test_epoch,
                name='test')
            test_acc = float(test_results['accuracy'])
            logger.debug("Test accuracy: %f", test_acc)
            results['test_accuracy'] = test_acc

            results[
                'training_time_in_hours'] = timer_manager.get_time_since_event(
                    'eval', 'start', units='hours')
        except:
            logger.info('Error during evaluation')
        finally:
            if self.delete_scratch_after_use:
                gcu.delete_bucket_folder(model_dir)
        return results

    def save_state(self, folderpath):
        pass

    def load_state(self, folderpath):
        pass
