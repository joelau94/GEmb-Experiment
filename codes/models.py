"""BiLSTM Models with GEMB"""
import abc
import cPickle as pkl

import numpy as np
import tensorflow as tf


class Embeddings(object):
  """Embeddings"""

  def __init__(self, vocab_size, embed_dim, reuse=None):

    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.reuse = reuse

    initializer = tf.zeros_initializer()
    with tf.variable_scope('Embeddings', reuse=self.reuse):
      self.emb = tf.get_variable(
          'emb',
          shape=(self.vocab_size, self.embed_dim),
          initializer=initializer
      )

  def load_pretrained_emb(self, embed_pret_file, w2i_pret_file):
    w2i_pret = pkl.load(open(w2i_pret_file, 'r'))
    self.emb_nparray = np.zeros([len(w2i_pret) + 1, self.embed_dim],
                                dtype=np.float32)

    for line in open(embed_pret_file, 'r'):
      we = line.strip().split()
      if we[0] in w2i_pret:
        self.emb_nparray[w2i_pret[we[0]]] = np.array(we[1:])

    with tf.variable_scope('Embeddings', reuse=self.reuse):
      self.emb_pret = tf.Variable(
          [0.0], trainable=False, name='emb_pret')
      self.emb_pret_placeholder = tf.placeholder(
          tf.float32, shape=self.emb_nparray.shape)
      self.embed_assign = tf.assign(
          self.emb_pret,
          self.emb_pret_placeholder,
          validate_shape=False)

  def init_pretrained_emb(self, sess):
    sess.run(self.embed_assign,
             feed_dict={self.emb_pret_placeholder: self.emb_nparray})

  def __call__(self, word_ids, word_ids_pret=None):
    embeddings = tf.nn.embedding_lookup(self.emb, word_ids)
    if word_ids_pret is not None:
      embeddings += tf.nn.embedding_lookup(self.emb_pret, word_ids_pret)
    return embeddings


class LSTMEncoder(object):
  """LSTMEncoder"""

  def __init__(self, hidden_dims, keep_prob=1.0, reuse=None):
    self.hidden_dims = hidden_dims
    self.keep_prob = keep_prob
    self.reuse = reuse

    with tf.variable_scope('LSTMEncoder', reuse=self.reuse):
      self.fw_step = self._step(hidden_dims)
      self.bw_step = self._step(hidden_dims)

  def _step(self, hidden_dims):
    with tf.variable_scope('LSTMEncoder', reuse=self.reuse):
      cells = [
          tf.contrib.rnn.LayerNormBasicLSTMCell(
              num_units=n, layer_norm=True, dropout_keep_prob=self.keep_prob)
          for n in hidden_dims
      ]
      return tf.contrib.rnn.MultiRNNCell(cells)

  def __call__(self, word_embeddings, sent_length):

    with tf.variable_scope('LSTMEncoder', reuse=self.reuse):

      outputs, states = tf.nn.bidirectional_dynamic_rnn(
          self.fw_step,
          self.bw_step,
          word_embeddings,
          sequence_length=sent_length,
          dtype=tf.float32,
          swap_memory=True
      )

      final_states = tf.concat([states[0][-1].h, states[1][-1].h], axis=-1)

      return final_states, outputs


class LSTMwithGEmb(object):
  """LSTMwithGEmb"""

  def __init__(self,
               bot_hidden_dim,
               hidden_dims=None,
               use_gemb=False,
               vocab_size=None,
               embed_dim=None,
               keep_prob=1.0,
               reuse=None):
    self.bot_hidden_dim = bot_hidden_dim
    self.hidden_dims = hidden_dims
    self.use_gemb = use_gemb
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.keep_prob = keep_prob
    self.reuse = reuse

  def __call__(self,
               word_embeddings,
               sent_length,
               oov_mask=None,
               embed_matrix=None):

    with tf.variable_scope('bottom_lstm', reuse=self.reuse):
      bot_lstm = LSTMEncoder([self.bot_hidden_dim],
                             keep_prob=self.keep_prob,
                             reuse=self.reuse)
      final_states, outputs = bot_lstm(word_embeddings, sent_length)

    if self.use_gemb:
      _, word_dist = self.get_word_dist(outputs, reuse=self.reuse)

      batch_size = tf.shape(word_dist)[0]
      max_length = tf.shape(word_dist)[1]
      vocab_size = tf.shape(word_dist)[2]

      # (batch, len, vocab) -> (batch * len, vocab)
      word_dist = tf.reshape(word_dist, [batch_size * max_length, -1])
      # (batch * len, vocab) x (vocab, embed_dim) -> (batch * len, embed_dim)
      gembeddings = tf.reshape(tf.matmul(word_dist, embed_matrix),
                               [batch_size, max_length, -1])
      oov_mask = tf.expand_dims(oov_mask, axis=-1)
      gembeddings = gembeddings * oov_mask + word_embeddings * (1. - oov_mask)
      gembeddings *= tf.expand_dims(
          tf.sequence_mask(sent_length, dtype=tf.float32),
          axis=-1)

      with tf.variable_scope('bottom_lstm', reuse=True):
        final_states, outputs = bot_lstm(gembeddings, sent_length)

    if self.hidden_dims:
      with tf.variable_scope('stack_lstm', reuse=self.reuse):
        stack_lstm = LSTMEncoder(self.hidden_dims,
                                 keep_prob=self.keep_prob,
                                 reuse=self.reuse)
        hidden_tape = tf.concat(outputs, axis=-1)
        final_states, outputs = stack_lstm(hidden_tape, sent_length)

    return final_states, outputs

  def gemb_train(self, word_ids, word_embeddings, sent_length):
    batch_size = tf.shape(word_ids)[0]
    max_length = tf.shape(word_ids)[1]

    with tf.variable_scope('bottom_lstm', reuse=True):
      bot_lstm = LSTMEncoder([self.bot_hidden_dim],
                             keep_prob=self.keep_prob,
                             reuse=True)
      final_states, outputs = bot_lstm(word_embeddings, sent_length)

    logits, _ = self.get_word_dist(outputs, reuse=True)
    word_one_hot = tf.one_hot(word_ids, self.vocab_size)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.reshape(word_one_hot, [batch_size * max_length, -1]),
        logits=tf.reshape(logits, [batch_size * max_length, -1])
    )
    loss = tf.reduce_sum(
        tf.reshape(loss, [batch_size, max_length]) *
        tf.sequence_mask(sent_length, dtype=tf.float32))

    return loss

  def get_word_dist(self, output_states, reuse=None):
    batch_size = tf.shape(output_states[0])[0]
    max_length = tf.shape(output_states[0])[1]
    hidden_dim = tf.shape(output_states[0])[2]
    zero_state = tf.zeros([batch_size, 1, hidden_dim])

    fw_states, bw_states = output_states
    fw_states = tf.concat([fw_states[:, 1:, :], zero_state], axis=1)
    bw_states = tf.concat([zero_state, bw_states[:, :-1, :]], axis=1)
    offset_states = tf.concat([fw_states, bw_states], axis=-1)

    with tf.variable_scope('gemb', reuse=reuse):
      logits = tf.layers.dense(offset_states, units=self.vocab_size)
      word_dist = tf.nn.softmax(logits, axis=-1)

    return logits, word_dist


class BiLstmModel(object):
  """BiLstmModel"""
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               vocab_size,
               embed_dim,
               hidden_dims,
               num_class,
               use_gemb=False,
               keep_prob=1.0,
               reuse=None):

    self.num_class = num_class
    self.use_gemb = use_gemb
    self.keep_prob = keep_prob
    self.reuse = reuse

    self.embedder = Embeddings(vocab_size, embed_dim, reuse=reuse)
    self.bilstm = LSTMwithGEmb(hidden_dims[0],
                               hidden_dims[1:],
                               use_gemb=use_gemb,
                               vocab_size=vocab_size,
                               embed_dim=embed_dim,
                               keep_prob=keep_prob,
                               reuse=reuse)

    self.word_ids = tf.placeholder(dtype=tf.int64, shape=(None, None))
    self.sent_length = tf.placeholder(dtype=tf.int64, shape=(None,))
    self.oov_mask = tf.placeholder(dtype=tf.float32, shape=(None, None))

  @abc.abstractmethod
  def __call__(self):
    pass

  def gemb_train(self):
    embeddings = self.embedder(self.word_ids)
    self.gemb_loss = self.bilstm.gemb_train(self.word_ids,
                                            embeddings,
                                            self.sent_length)
    return self.gemb_loss


class SeqTaggingModel(BiLstmModel):
  """SeqTaggingModel"""

  def __call__(self):

    self.labels = tf.placeholder(dtype=tf.int64, shape=(None, None))
    embeddings = self.embedder(self.word_ids)
    if self.use_gemb:
      _, outputs = self.bilstm(embeddings,
                               self.sent_length,
                               oov_mask=self.oov_mask,
                               embed_matrix=self.embedder.emb)
    else:
      _, outputs = self.bilstm(embeddings, self.sent_length)

    hidden_tape = tf.concat(outputs, axis=-1)

    with tf.variable_scope('output_mlp', reuse=self.reuse):
      feats = tf.nn.dropout(hidden_tape, keep_prob=self.keep_prob)
      scores = tf.layers.dense(feats, units=self.num_class, name='scores')

    self.probabilities = tf.nn.softmax(scores, axis=-1)
    self.predictions = tf.argmax(self.probabilities, axis=-1)

    masks = tf.sequence_mask(self.sent_length, dtype=tf.float32)

    self.loss = tf.contrib.seq2seq.sequence_loss(scores, self.labels, masks)
    correct_count = tf.reduce_sum(
        tf.cast(tf.equal(self.predictions, self.labels),
                dtype=tf.float32) * masks)
    total_count = tf.reduce_sum(masks)

    return self.loss, correct_count, total_count


class SeqClassifierModel(BiLstmModel):
  """SeqClassifierModel"""

  def __call__(self):
    self.labels = tf.placeholder(dtype=tf.int64, shape=(None,))
    embeddings = self.embedder(self.word_ids)
    if self.use_gemb:
      final_states, _ = self.bilstm(embeddings,
                                    self.sent_length,
                                    oov_mask=self.oov_mask,
                                    embed_matrix=self.embedder.emb)
    else:
      final_states, _ = self.bilstm(embeddings, self.sent_length)

    with tf.variable_scope('output_mlp', reuse=self.reuse):
      feats = tf.nn.dropout(final_states, keep_prob=self.keep_prob)
      scores = tf.layers.dense(feats, units=self.num_class, name='scores')

    self.probabilities = tf.nn.softmax(scores, axis=-1)
    self.predictions = tf.argmax(self.probabilities, axis=-1)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.labels, logits=scores)

    self.loss = tf.reduce_mean(losses)
    correct_count = tf.reduce_sum(
        tf.cast(tf.math.equal(self.predictions, self.labels),
                dtype=tf.float32))
    total_count = tf.reduce_sum(masks)

    return self.loss, correct_count, total_count
