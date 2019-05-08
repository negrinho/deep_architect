from __future__ import print_function

import os
import sys
import gc
import subprocess
import random
import logging
from six.moves import range

import tensorflow as tf
import numpy as np

from deep_architect.contrib.misc.datasets.cifar10_tf import Cifar10DataSet
import deep_architect.core as co
import deep_architect.utils as ut
import dev.helpers.tfeager as htfe
# import hooks
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = logging.getLogger(__name__)

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3


def setRecompile(output_lst, recompile):

    def fn(mx):
        mx._is_compiled = not recompile

    co.traverse_backward(output_lst, fn)
    logger.debug('set_recompile')


def input_fn(mode, data_dir, batch_size=128, train=True):
    return Cifar10DataSet(data_dir, subset=mode,
                          use_distortion=train).make_batch(batch_size)


def get_empty_bucket_folder(folder):
    folder_name = ''
    while True:
        num = random.randint(0, sys.maxsize)
        folder_name = os.path.join(folder, 'eval' + str(num))
        try:
            subprocess.check_call(['gsutil', '-q', 'stat', folder_name + '/**'])
        except subprocess.CalledProcessError:
            break
    return folder_name


def delete_bucket_folder(folder):
    try:
        subprocess.check_call(['gsutil', '-m', 'rm', folder + '/**'])
    except:
        pass


class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):

    def __init__(self, estimator, input_fn, steps_per_val_epoch):
        self.estimator = estimator
        self.input_fn = input_fn
        self.steps_per_val_epoch = steps_per_val_epoch

    def after_save(self, session, global_step):
        self.estimator.evaluate(self.input_fn, steps=self.steps_per_val_epoch)


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


class AdvanceClassifierEvaluator:

    def __init__(self,
                 data_dir,
                 tpu_name,
                 max_num_training_epochs=200,
                 stop_patience=20,
                 optimizer_type='sgd_mom',
                 batch_size=256,
                 lr_decay_method='constant',
                 init_lr=.001,
                 lr_decay_value=.97,
                 lr_num_epochs_per_decay=2.4,
                 lr_warmup_epochs=3.0,
                 weight_decay=.0005,
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
        # total_steps = int(
        #     self.max_num_training_epochs * self.num_examples / self.batch_size)
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
        logger.debug('In Evaluator')
        if state is not None and 'model_dir' in state:
            model_dir = state['model_dir']
        else:
            model_dir = get_empty_bucket_folder(self.base_dir)
            save_fn({'model_dir': model_dir})
        logger.info('Using folder %s for evaluation', model_dir)

        def metric_fn(labels, predictions):
            return {'accuracy': tf.metrics.accuracy(labels, predictions)}

        def model_fn(features, labels, mode, params):
            setRecompile(outputs.values(), True)
            gc.collect()
            htfe.setTraining(outputs.values(),
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
                    tf.math.divide(
                        tf.cast(step, tf.float32),
                        float(self.steps_per_epoch *
                              self.max_num_training_epochs))
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
                onehot_labels=onehot_labels, logits=logits)
            aux_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=aux_logits,
                weights=.5) if 'Out1' in outputs else 0
            # tf.reduce_sum(
            #     tf.nn.sparse_softmax_cross_entropy_with_logits(
            #         logits=logits, labels=labels))
            loss = unreg_loss + l2_loss + aux_loss
            if mode == tf.estimator.ModeKeys.EVAL:
                # return tf.estimator.EstimatorSpec(mode,
                #                                   loss=loss,
                #                                   eval_metric_ops=metric_fn(
                #                                       labels,
                #                                       predicted_classes))
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
            # optimizer = tf.train.RMSPropOptimizer(learning_rate,
            #                                       momentum=.9,
            #                                       epsilon=1.0)
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=.9)
            host_fn = None
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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

                # train_op = optimizer.minimize(
                #     loss, global_step=tf.train.get_or_create_global_step())
            # return tf.estimator.EstimatorSpec(
            #     mode,
            #     loss=loss,
            #     train_op=train_op,
            # )
            return tf.contrib.tpu.TPUEstimatorSpec(mode,
                                                   loss=loss,
                                                   train_op=train_op,
                                                   host_call=host_fn)

        # NUM_GPUS = 2
        # strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
        # config = tf.estimator.RunConfig(train_distribute=strategy)
        # gpu_ops = tf.GPUOptions(allow_growth=True)
        # config = tf.ConfigProto(gpu_options=gpu_ops)
        # run_config = tf.estimator.RunConfig(
        #     model_dir=model_dir, session_config=config)

        my_project_name = subprocess.check_output(
            ['gcloud', 'config', 'get-value', 'project'])
        my_zone = subprocess.check_output(
            ['gcloud', 'config', 'get-value', 'compute/zone'])
        cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            self.tpu_name.split(','), zone=my_zone, project=my_project_name)
        run_config = tf.contrib.tpu.RunConfig(
            cluster=cluster_resolver,
            model_dir=model_dir,
            save_checkpoints_steps=self.max_num_training_epochs *
            self.steps_per_epoch // 3,
            keep_checkpoint_max=1,
            log_step_count_steps=self.steps_per_epoch,
            # session_config=tf.ConfigProto(
            #     allow_soft_placement=True, log_device_placement=True),
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.steps_per_epoch, num_shards=8),
        )

        try:
            estimator = tf.contrib.tpu.TPUEstimator(
                model_fn=model_fn,
                config=run_config,
                use_tpu=self.use_tpu,
                train_batch_size=self.batch_size,
                eval_batch_size=self.batch_size,
                predict_batch_size=self.batch_size,
                params={})
            # estimator = tf.estimator.Estimator(model_fn=model_fn,
            #                                    model_dir='temp/')
            seqs = ut.SequenceTracker(abort_if_different_lengths=True)

            best_val_acc = -np.inf
            stop_counter = self.stop_patience
            timer_manager = ut.TimerManager()
            timer_manager.create_timer('eval')

            train_fn = lambda params: input_fn(
                'train',
                self.data_dir,
                #    batch_size=params['batch_size'],
                batch_size=self.batch_size,
                train=True)
            val_fn = lambda params: input_fn(
                'validation',
                self.data_dir,
                #  batch_size=params['batch_size'],
                batch_size=self.batch_size,
                train=False)
            # train_hook = hooks.InMemoryEvaluatorHook(
            #     estimator,
            #     val_fn,
            #     steps=self.steps_per_val_epoch,
            #     every_n_iter=1)
            # estimator.train(
            #     input_fn=train_fn,
            #     steps=1,
            # )
            # estimator.evaluate(
            #     input_fn=val_fn,
            #     steps=1,
            # )
            try:
                estimator.train(input_fn=train_fn,
                                max_steps=self.steps_per_epoch *
                                self.max_num_training_epochs)
            except (tf.train.NanLossDuringTrainingError,
                    tf.errors.InvalidArgumentError):
                logger.warning(
                    'Architecture in %s received nan loss in training',
                    model_dir)

            # print(
            #     np.sum([
            #         np.prod(v.get_shape().as_list())
            #         for v in tf.trainable_variables()
            #     ]))
            # hooks=[
            #     train_hook,
            # ])
            # epochs = 0
            # while epochs < self.max_num_training_epochs:
            #     logger.info('epoch %d', epochs)
            #     epochs_to_train = min(self.epochs_between_evals,
            #                           self.max_num_training_epochs - epochs)
            #     estimator.train(
            #         input_fn=train_fn,
            #         steps=self.steps_per_epoch * epochs_to_train)
            #     epochs += epochs_to_train
            #     eval_results = estimator.evaluate(
            #         input_fn=val_fn, steps=self.steps_per_val_epoch)
            #     logger.info('Results epoch %d: %s', epochs, eval_results)
            #     # for epoch in range(self.max_num_training_epochs):
            #     # train_spec = tf.estimator.TrainSpec(input_fn=train_fn)
            #     # eval_spec = tf.estimator.EvalSpec(input_fn=val_fn)
            #     # val_fn = tf.estimator.inputs.numpy_input_fn(self.X_val, self.y_val, shuffle=False, num_threads=8)
            #     # train_spec = tf.estimator.TrainSpec(
            #     #     input_fn=train_fn, max_steps=1000)
            #     # eval_spec = tf.estimator.EvalSpec(input_fn=val_fn)

            #     # eval_results = tf.estimator.train_and_evaluate(
            #     #     estimator, train_spec, eval_spec)
            #     # print('Training')
            #     # estimator.train(
            #     #     input_fn=train_fn,
            #     #     steps=self.steps_per_epoch * self.max_num_training_epochs,
            #     # )
            #     # print('Evaluating')
            #     # print('Evaluating done')
            #     # print('\n\nTraining and Evaluating')
            #     # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

            #     # early stopping
            #     val_acc = float(eval_results['accuracy'])

            #     # Display logs per epoch step
            #     # if self.log_output_to_terminal and epoch % self.display_step == 0:
            #     #     print("time:", "%7.1f" % timer_manager.get_time_since_event(
            #     #         'eval', 'start'), "epoch:", '%04d' % (epoch + 1),
            #     #           "validation loss:", "{:.9f}".format(
            #     #               eval_results['loss']), "validation_accuracy:",
            #     #           "%.5f" % val_acc)

            #     d = {
            #         'validation_accuracy':
            #         val_acc,
            #         'validation_loss':
            #         float(eval_results['loss']),
            #         'epoch_number':
            #         epochs,
            #         'time_in_minutes':
            #         timer_manager.get_time_since_event(
            #             'eval', 'start', units='minutes'),
            #     }
            #     seqs.append(d)

            #     # update the patience counters.
            #     if best_val_acc < val_acc:
            #         best_val_acc = val_acc
            #         # reinitialize all the counters.
            #         stop_counter = self.stop_patience
            #     else:
            #         stop_counter -= 1
            #         if stop_counter == 0:
            #             break

            logger.debug("Optimization Finished!")

            val_fn = lambda params: input_fn('validation',
                                             self.data_dir,
                                             batch_size=params['batch_size'],
                                             train=False)

            timer_manager.tick_timer('eval')
            eval_results = estimator.evaluate(input_fn=val_fn,
                                              steps=self.steps_per_val_epoch)
            t_infer = (
                timer_manager.get_time_since_last_tick('eval', 'miliseconds') /
                Cifar10DataSet.num_examples_per_epoch('validation'))

            val_acc = float(eval_results['accuracy'])
            logger.debug("Validation accuracy: %f", val_acc)

            seqs_dict = seqs.get_dict()
            results = {
                'validation_accuracy': val_acc,
                'num_parameters': self.num_parameters,
                'inference_time_per_example_in_miliseconds': t_infer,
                # 'num_training_epochs': seqs_dict['epoch_number'],
                'sequences': seqs_dict
            }
            test_fn = lambda params: input_fn('eval',
                                              self.data_dir,
                                              batch_size=params['batch_size'],
                                              train=False)

            test_results = estimator.evaluate(input_fn=test_fn,
                                              steps=self.steps_per_test_epoch,
                                              name='test')
            test_acc = float(test_results['accuracy'])
            logger.debug("Test accuracy: %f", test_acc)
            results['test_accuracy'] = test_acc

            results[
                'training_time_in_hours'] = timer_manager.get_time_since_event(
                    'eval', 'start', units='hours')
        except:
            raise
        finally:
            # self.num_archs += 1
            if self.delete_scratch_after_use:
                delete_bucket_folder(model_dir)
        return results

    def save_state(self, folder):
        pass

    def load_state(self, folder):
        pass