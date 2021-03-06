import os
import cPickle as pkl
import math

import numpy as np
import tensorflow as tf

import data
import models


class Config(object):
  """Config"""

  def __init__(self):
    self.config = {
        'train_data_file': '../data/train.pkl',
        'dev_data_file': '../data/dev.pkl',
        'test_data_file': '../data/test.pkl',

        'task': 'tagging',
        'keep_prob': 0.9,

        'vocab_size': 30000,
        'embed_dim': 300,
        'hidden_dims': [256, 256],
        'num_class': 50,
        'early_stop': False,

        'seed': 23,
        'lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.99,
        'eps': 1e-8,
        'clip_norm': 1.0,
        'global_norm': 5.0,

        'ckpt': '../models/model',
        'max_ckpts': 20,
        'batch_size': 64,
        'max_steps': 1000000,
        'gemb_steps': 1000000,
        'print_interval': 50,
        'save_interval': 1000
    }

  def save(self, filename):
    pkl.dump(self.config, open(filename, 'wb'))

  def load(self, filename):
    self.config = pkl.load(open(filename, 'rb'))


class Experiment(object):
  """Experiment"""

  def __init__(self, config):
    self.config = config
    self.patience_cnt = 0
    self.max_acc = 0.

  def train(self, use_gemb=False):
    train_data = data.Dataset(self.config['train_data_file'],
                              shuffle=True,
                              seed=self.config['seed'])
    if self.config['task'] == 'tagging':
      self.model_class = models.SeqTaggingModel
    elif self.config['task'] == 'classfication':
      self.model_class = models.SeqClassifierModel

    dev_data = data.Dataset(self.config['dev_data_file'],
                            shuffle=False)

    train_graph = tf.Graph()

    with tf.Session(graph=train_graph) as sess:
      model = self.model_class(self.config['vocab_size'],
                               self.config['embed_dim'],
                               self.config['hidden_dims'],
                               self.config['num_class'],
                               keep_prob=self.config['keep_prob'],
                               reuse=None)

      loss, correct_count, total_count = model()

      optimizer = tf.train.AdamOptimizer(
          learning_rate=self.config['lr'],
          beta1=self.config['beta1'],
          beta2=self.config['beta2'],
          epsilon=self.config['eps'])

      # Clip by value.
      gradients = optimizer.compute_gradients(loss)
      # Clipping by global norm scales by 0.2 which results in NaNs (not sure why?), this has been changed to scale tensor to a range between -1 and 1
      gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      train_op = optimizer.apply_gradients(gradients,
                                           global_step=self.global_step)

      # Clip by global norm.
      #model_params = tf.trainable_variables()
      #grad = tf.gradients(loss, model_params)
      #clip_grad, _ = tf.clip_by_global_norm(
      #    grad,
      #    self.config['clip_norm'],
      #    use_norm=self.config['global_norm'])

      #self.global_step = tf.Variable(0, name='global_step', trainable=False)
      #train_op = optimizer.apply_gradients(zip(clip_grad, model_params),
      #                                     global_step=self.global_step)

      sess.run(tf.global_variables_initializer())

      train_saver = tf.train.Saver(max_to_keep=self.config['max_ckpts'])
      ckpt_dir = os.path.dirname(self.config['ckpt'])
      if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)
      ckpt = tf.train.latest_checkpoint(ckpt_dir)
      if ckpt:
        train_saver.restore(sess, ckpt)

      print('Training ...')
      global_steps = sess.run(self.global_step)
      while global_steps <= self.config['max_steps']:

        X, Y, sent_length, oov_mask = \
            train_data.get_next(self.config['batch_size'])

        if not use_gemb:
          oov_mask = np.zeros_like(oov_mask, dtype=np.float32)

        _, step_loss, global_steps = sess.run(
            [train_op, loss, self.global_step],
            feed_dict={
                model.word_ids: X,
                model.labels: Y,
                model.sent_length: sent_length,
                model.oov_mask: oov_mask
            })

        if global_steps % self.config['print_interval'] == 0:
          print('Step {}: Loss = {}'.format(global_steps, step_loss))

        if global_steps % self.config['save_interval'] == 0:
          if not self._validate_save(model, sess, train_saver, dev_data, use_gemb,
                                     global_steps, correct_count, total_count):
            break

      # Exit train loop

  def train_gemb(self):
    train_data = data.Dataset(self.config['train_data_file'],
                              shuffle=True,
                              seed=self.config['seed'])
    if self.config['task'] == 'tagging':
      self.model_class = models.SeqTaggingModel
    elif self.config['task'] == 'classfication':
      self.model_class = models.SeqClassifierModel

    dev_data = data.Dataset(self.config['dev_data_file'],
                            shuffle=False)

    train_gemb_graph = tf.Graph()

    with tf.Session(graph=train_gemb_graph) as sess:
      model = self.model_class(self.config['vocab_size'],
                               self.config['embed_dim'],
                               self.config['hidden_dims'],
                               self.config['num_class'],
                               keep_prob=self.config['keep_prob'],
                               reuse=None)

      _, correct_count, total_count = model()
      gemb_loss = model.gemb_train()

      optimizer = tf.train.AdamOptimizer(
          learning_rate=self.config['lr'],
          beta1=self.config['beta1'],
          beta2=self.config['beta2'],
          epsilon=self.config['eps'])
      gemb_params = [var for var in tf.trainable_variables()
                     if 'gemb' in var.op.name]
      grad = tf.gradients(gemb_loss, gemb_params)
      clip_grad, _ = tf.clip_by_global_norm(
          grad,
          self.config['clip_norm'],
          use_norm=self.config['global_norm'])

      # self.global_step = tf.Variable(0, name='global_step', trainable=False)
      train_op = optimizer.apply_gradients(zip(clip_grad, gemb_params))

      sess.run(tf.global_variables_initializer())

      train_saver = tf.train.Saver(max_to_keep=self.config['max_ckpts'])
      ckpt_dir = os.path.dirname(self.config['ckpt'])
      ckpt = tf.train.latest_checkpoint(ckpt_dir)
      train_saver.restore(sess, ckpt)

      print('Training ...')
      for step in range(self.config['gemb_steps']):
        X, Y, sent_length, oov_mask = \
            train_data.get_next(self.config['batch_size'])

        # train all words
        oov_mask = np.ones_like(oov_mask, dtype=np.float32)

        _, step_loss = sess.run(
            [train_op, gemb_loss],
            feed_dict={
                model.word_ids: X,
                model.sent_length: sent_length
            })

        if step % self.config['print_interval'] == 0:
          print('Step {}: Loss = {}'.format(step, step_loss))

        if step % self.config['save_interval'] == 0:
          train_saver.save(sess, self.config['ckpt'])
          # validation
          batch_num = int(math.floor(len(dev_data.records) /
                                     self.config['batch_size']))
          corr, total = 0, 0
          for _ in range(batch_num):
            X, Y, sent_length, oov_mask = \
                dev_data.get_next(self.config['batch_size'])
            c, t = sess.run(
                [correct_count, total_count],
                feed_dict={
                    model.word_ids: X,
                    model.labels: Y,
                    model.sent_length: sent_length,
                    model.oov_mask: oov_mask
                })
            corr += c
            total += t
          print('Step {}: Acc = {}'.format(step, float(corr) / total))

      # Exit train loop

      train_saver.save(sess, self.config['ckpt'])
      # validation
      batch_num = int(math.floor(len(dev_data.records) /
                                 self.config['batch_size']))
      corr, total = 0, 0
      for _ in range(batch_num):
        X, Y, sent_length, oov_mask = \
            dev_data.get_next(self.config['batch_size'])
        c, t = sess.run(
            [correct_count, total_count],
            feed_dict={
                model.word_ids: X,
                model.labels: Y,
                model.sent_length: sent_length,
                model.oov_mask: oov_mask
            })
        corr += c
        total += t
      print('Acc = {}'.format(float(corr) / total))

  def test(self, use_gemb=False):
    test_data = data.Dataset(self.config['test_data_file'], shuffle=False)
    if self.config['task'] == 'tagging':
      self.model_class = models.SeqTaggingModel
    elif self.config['task'] == 'classfication':
      self.model_class = models.SeqClassifierModel

    test_graph = tf.Graph()

    with tf.Session(graph=test_graph) as sess:
      model = self.model_class(self.config['vocab_size'],
                               self.config['embed_dim'],
                               self.config['hidden_dims'],
                               self.config['num_class'],
                               keep_prob=self.config['keep_prob'],
                               reuse=None)

      loss, correct_count, total_count = model()

      sess.run(tf.global_variables_initializer())

      saver = tf.train.Saver(max_to_keep=self.config['max_ckpts'])
      ckpt_dir = os.path.dirname(self.config['ckpt'])
      ckpt = tf.train.latest_checkpoint(ckpt_dir)
      saver.restore(sess, ckpt)

      print('Testing ...')
      self._validate(model, sess, test_data, use_gemb, correct_count, total_count)


  def _validate(self, model, sess, data, use_gemb, correct_count, total_count):
    batch_num = int(math.floor(len(data.records) /
                               self.config['batch_size']))
    corr, total = 0, 0
    for _ in range(batch_num):
      X, Y, sent_length, oov_mask = \
          data.get_next(self.config['batch_size'])
      if not use_gemb:
        oov_mask = np.zeros_like(oov_mask, dtype=np.float32)

      c, t = sess.run(
          [correct_count, total_count],
          feed_dict={
              model.word_ids: X,
              model.labels: Y,
              model.sent_length: sent_length,
              model.oov_mask: oov_mask
          })
      corr += c
      total += t
    acc = float(corr)/total
    return acc


  def _validate_save(self, model, sess, train_saver, data, use_gemb, global_steps,
                     correct_count, total_count):
    # Validation.
    acc = self._validate(model, sess, data, use_gemb, correct_count, total_count)
    print('Step: {} Acc = {}'.format(global_steps, acc))

    if self.config['early_stop'] and acc < self.max_acc:
      self.patience_cnt += 1
      if self.patience_cnt == self.config['patience']:
        print('Accuracy has been lower than the max accuracy of {} '
              'for {} intervals.'.format(self.max_acc, self.patience_cnt))
        return False
    else:
      self.max_acc = max(acc, self.max_acc)
      self.patience_cnt = 0
      train_saver.save(sess, self.config['ckpt'],
                       global_step=global_steps)

    return True
