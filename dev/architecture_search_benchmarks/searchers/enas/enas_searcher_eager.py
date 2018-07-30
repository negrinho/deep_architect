from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
import sys
import os
import time

import numpy as np
import tensorflow as tf
tfe = tf.contrib.eager

from dev.architecture_search_benchmarks.searchers.enas.enas_common_ops import stack_lstm

from darch.searchers import (Searcher, unset_hyperparameter_iterator,
                             random_specify_hyperparameter, random_specify)

class ENASEagerSearcher(Searcher):
    def __init__(self,
                 search_space_fn,
                 num_layers=12,
                 num_branches=6,
                 out_filters=48,
                 lstm_size=32,
                 lstm_num_layers=2,
                 lstm_keep_prob=1.0,
                 tanh_constant=1.5,
                 learning_rate=1e-3,
                 entropy_weight=.1,
                 bl_dec=0.999,
                 skip_target=0.4,
                 skip_weight=0.8,
                 name="controller",
                 *args,
                 **kwargs):
        Searcher.__init__(self, search_space_fn)

        print("-" * 80)
        print("Building ConvController")

        self.num_layers = num_layers
        self.num_branches = num_branches
        self.out_filters = out_filters

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_keep_prob = lstm_keep_prob
        self.tanh_constant = tanh_constant
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight
        self.bl_dec = bl_dec

        self.skip_target = skip_target
        self.skip_weight = skip_weight

        self.name = name

        self._create_params()
        self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.0, epsilon=1e-3,
                                    use_locking=True)
        self.train_step = tfe.Variable(
            0, dtype=tf.int32, trainable=False, name="train_step")

    def sample(self):
        arc = self._sample()
        idx = 0
        hyp_values = {}
        for i in range(1, self.num_layers + 1):
            hyp_values['op_' + str(i)] = arc[idx]
            idx += 1
            for j in range(i - 1):
                hyp_values['H.skip_' + str(j) + '_' + str(i) + '-0'] = arc[idx]
                idx += 1

        inputs, outputs, hs = self.search_space_fn()
        vs = []
        for i, h in enumerate(unset_hyperparameter_iterator(list(outputs.values()), list(hs.values()))):
            if h.get_name() in hyp_values:
                v = h.vs[hyp_values[h.get_name()]]
                h.set_val(v)
                vs.append(v)
            else:
                v = random_specify_hyperparameter(h)
                vs.append(v)
        return inputs, outputs, hs, vs, {'arc': arc}

    def update(self, val, cfg_d):
        if val != -1:
            loss = self._train(val, cfg_d['arc'])
            if self.train_step % 1 == 0:
                print("Step: %d,    Reward: %f,    Loss:%f" % (self.train_step, val, loss.numpy()))
            

    def save_state(self, folder_name):
        pass

    def load(self, state):
        pass

    def _create_params(self):
        self.variables = []
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            with tf.device('/gpu:0'):
                with tf.variable_scope("lstm"):
                    self.w_lstm = []
                    for layer_id in range(self.lstm_num_layers):
                        with tf.variable_scope("layer_{}".format(layer_id)):
                            w = tfe.Variable(initializer([2 * self.lstm_size, 4 * self.lstm_size]),
                                name="w")
                            self.w_lstm.append(w)
                    self.variables.extend(self.w_lstm)

                self.g_emb = tfe.Variable(initializer([1, self.lstm_size]), name="g_emb")
                with tf.variable_scope("emb"):
                    self.w_emb = tfe.Variable(
                        initializer([self.num_branches, self.lstm_size]), name="w")
                with tf.variable_scope("softmax"):
                    self.w_soft = tfe.Variable(
                        initializer([self.lstm_size, self.num_branches]), name="w")

                with tf.variable_scope("attention"):
                    self.w_attn_1 = tfe.Variable(
                        initializer([self.lstm_size, self.lstm_size]), name="w_1")
                    self.w_attn_2 = tfe.Variable(
                        initializer([self.lstm_size, self.lstm_size]), name="w_2")
                    self.v_attn = tfe.Variable(initializer([self.lstm_size, 1]), name="v")
                self.variables.extend([
                    self.g_emb, self.w_emb, self.w_soft, self.w_attn_1, self.w_attn_2,
                    self.v_attn])

                self.baseline = tfe.Variable(0.0, dtype=tf.float32, trainable=False)

    def _sample(self):
        """Build the sampler ops and the log_prob ops."""
        print("-" * 80)
        print("Build controller sampler")
        anchors = []
        anchors_w_1 = []

        arc_seq = []

        with tf.device('/gpu:0'):
            prev_c = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
                    range(self.lstm_num_layers)]
            prev_h = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
                    range(self.lstm_num_layers)]
            inputs = self.g_emb
            for layer_id in range(self.num_layers):
                next_c, next_h = stack_lstm(
                    inputs, prev_c, prev_h, self.w_lstm)
                prev_c, prev_h = next_c, next_h
                logit = tf.matmul(next_h[-1], self.w_soft)
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * tf.tanh(logit)
                branch_id = tf.multinomial(logit, 1)
                branch_id = tf.to_int32(branch_id)
                branch_id = tf.reshape(branch_id, [1])
                arc_seq.append(branch_id)
                inputs = tf.nn.embedding_lookup(self.w_emb, branch_id)

                next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                prev_c, prev_h = next_c, next_h

                if layer_id > 0:
                    query = tf.concat(anchors_w_1, axis=0)
                    query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))
                    query = tf.matmul(query, self.v_attn)
                    logit = tf.concat([-query, query], axis=1)
                    if self.tanh_constant is not None:
                        logit = self.tanh_constant * tf.tanh(logit)

                    skip = tf.multinomial(logit, 1)
                    skip = tf.to_int32(skip)
                    skip = tf.reshape(skip, [layer_id])
                    arc_seq.append(skip)

                    skip = tf.to_float(skip)
                    skip = tf.reshape(skip, [1, layer_id])
                    inputs = tf.matmul(skip, tf.concat(anchors, axis=0))
                    inputs /= (1.0 + tf.reduce_sum(skip))
                else:
                    inputs = self.g_emb

                anchors.append(next_h[-1])
                anchors_w_1.append(tf.matmul(next_h[-1], self.w_attn_1))

            arc_seq = tf.concat(arc_seq, axis=0)
            sample_arc = tf.reshape(arc_seq, [-1])
        return sample_arc.numpy().tolist()

    def _train(self, valid_acc, prev_arc):
        with tf.device('/gpu:0'):
            with tf.GradientTape() as tape:    
                reward = valid_acc
                
                anchors = []
                anchors_w_1 = []

                entropys = []
                log_probs = []
                skip_penaltys = []

                prev_c = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
                        range(self.lstm_num_layers)]
                prev_h = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
                        range(self.lstm_num_layers)]
                inputs = self.g_emb
                skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target],
                                        dtype=tf.float32)
                idx = 0
                for layer_id in range(self.num_layers):
                    next_c, next_h = stack_lstm(
                        inputs, prev_c, prev_h, self.w_lstm)
                    prev_c, prev_h = next_c, next_h
                    logit = tf.matmul(next_h[-1], self.w_soft)
                    if self.tanh_constant is not None:
                        logit = self.tanh_constant * tf.tanh(logit)
                    branch_id = prev_arc[idx]
                    branch_id = tf.to_int32(branch_id)
                    branch_id = tf.reshape(branch_id, [1])
                    idx += 1

                    log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logit, labels=branch_id)
                    log_probs.append(log_prob)
                    entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
                    entropys.append(entropy)
                    inputs = tf.nn.embedding_lookup(self.w_emb, branch_id)

                    next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                    prev_c, prev_h = next_c, next_h

                    if layer_id > 0:
                        query = tf.concat(anchors_w_1, axis=0)
                        query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))
                        query = tf.matmul(query, self.v_attn)
                        logit = tf.concat([-query, query], axis=1)
                        if self.tanh_constant is not None:
                            logit = self.tanh_constant * tf.tanh(logit)

                        skip = prev_arc[idx: idx + layer_id]
                        skip = tf.to_int32(skip)
                        skip = tf.reshape(skip, [layer_id])
                        idx += layer_id

                        skip_prob = tf.sigmoid(logit)
                        kl = skip_prob * tf.log(old_div(skip_prob, skip_targets))
                        kl = tf.reduce_sum(kl)
                        skip_penaltys.append(kl)

                        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=logit, labels=skip)
                        log_probs.append(tf.reduce_sum(log_prob, keep_dims=True))

                        skip = tf.to_float(skip)
                        skip = tf.reshape(skip, [1, layer_id])
                        inputs = tf.matmul(skip, tf.concat(anchors, axis=0))
                        inputs /= (1.0 + tf.reduce_sum(skip))
                    else:
                        inputs = self.g_emb

                    anchors.append(next_h[-1])
                    anchors_w_1.append(tf.matmul(next_h[-1], self.w_attn_1))

                # self.sample_arc = tf.Print(self.sample_arc, [self.sample_arc], message="AFTER")
                sample_log_prob = tf.reduce_sum(log_probs)

                entropys = tf.stack(entropys)
                sample_entropy = tf.reduce_sum(entropys)
                
                skip_penaltys = tf.stack(skip_penaltys)
                skip_penaltys = tf.reduce_mean(skip_penaltys)
                
                reward += self.entropy_weight * sample_entropy
                
                self.baseline.assign_sub((1 - self.bl_dec) * (self.baseline - reward))

                loss = sample_log_prob * (reward - self.baseline)
                loss += self.skip_weight * skip_penaltys

            grads = tape.gradient(loss, self.variables)
            self.opt.apply_gradients(list(zip(grads, self.variables)), global_step=self.train_step)
        return loss
