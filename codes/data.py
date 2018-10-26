from collections import Counter, defaultdict
import cPickle as pkl
import itertools
import random

import numpy as np


class Dataset(object):
  """Dataset"""

  def __init__(self, datafile, shuffle=False, seed=None):
    dataset = pkl.load(open(datafile, 'rb'))
    # meta: dict; records: list(list(x: list, y: int/list))
    self.meta, self.records = dataset['meta'], dataset['records']
    self.shuffle = shuffle

    random.seed(seed)
    self.order = list(range(len(self.records)))
    if self.shuffle:
      random.shuffle(self.order)
    self.cursor = 0

    print('{} records loaded.'.format(len(self.records)))

  def get_next(n=1):
    if n > len(self.records):
      print('Batch size must be smaller than dataset size.')
      exit(0)
    start = self.cursor
    stop = (self.cursor + n) % len(self.records)
    if stop > start:
      batch = [self.records[i] for i in self.order[start:stop]]
      self.cursor = stop
    else:
      batch = [self.records[i] for i in self.order[start:]]
      random.shuffle(self.order)
      self.cursor = 0

    xs, ys = list(zip(*batch))
    sent_length = [len(x) for x in xs]
    max_length = max(sent_length)
    xs = list(map(lambda x: x + [0] * max_length - len(x), xs))
    if self.meta['task'] == 'tagging':  # tagging / classification
      ys = list(map(lambda y: y + [0] * max_length - len(y), ys))
    oov_mask = [[1. if i == self.meta['oov_id'] else 0.
                 for i in x] for x in xs]

    X = np.array(xs, dtype=np.int32)
    Y = np.array(ys, dtype=np.int32)
    sent_length = np.array(sent_length, dtype=np.int32)
    oov_mask = np.array(oov_mask, dtype=np.float32)

    return X, Y, sent_length, oov_mask


def parse_txt(txtfile, task):
  raw = map(lambda l: l.strip().split('|||'), open(txtfile, 'r').readlines())
  records = []

  if task == 'tagging':
    for r in raw:
      if not len(r) == 2:
        continue
      x = r[0].strip().split()
      y = r[1].strip().split()
      if not len(x) == len(y):
        continue
      records.append([x, y])
  elif task == 'classification':
    for r in raw:
      if not len(r) == 2:
        continue
      x = r[0].strip().split()
      y = r[1].strip()
      if not len(y) == 1:
        continue
      records.append([x, y])

  return records


def build_dicts(records, output_file, min_freq=2):
  word_count = Counter(itertools.chain(*list(zip(*records))[0]))
  tag_set = set(itertools.chain(*list(zip(*records))[0]))
  dicts = {}
  meta = {}

  dicts['i2w'] = ['_UNK_'] + [k for k, v in word_count if v >= min_freq]
  dicts['w2i'] = defaultdict(int)
  dicts['w2i'].update({w: i for i, w in enumerate(dicts['i2w'])})

  dicts['i2t'] = list(tag_set)
  dicts['t2i'] = {w: i for i, w in enumerate(dicts['i2t'])}

  pkl.dump(dicts, open(output_file, 'wb'))

  meta['oov_id'] = 0
  meta['vocab_size'] = len(dicts['i2w'])
  meta['num_class'] = len(dicts['i2t'])
  print(meta)

  return dicts, meta


def preprocess_train(task, raw_data_file, dictfile, datafile, min_freq=2):
  records = parse_txt(raw_data_file, task)
  dicts, meta = build_dicts(records, dictfile, min_freq)

  meta.update({'task': task})

  if task == 'tagging':
    records = list(map(
        lambda r: [list(map(lambda w: dicts['w2i'][w], r[0])),
                   list(map(lambda t: dicts['t2i'][t], r[1]))]
        records
    ))
  elif task == 'classification':
    records = list(map(
        lambda r: [list(map(lambda w: dicts['w2i'][w], r[0])),
                   dicts['t2i'][t]]
        records
    ))

  pkl.dump({'meta': meta, 'records': records},
           open(datafile, 'wb'))


def preprocess_dev_test(task, raw_data_file, dictfile, datafile):
  records = parse_txt(raw_data_file, task)
  dicts = pkl.load(open(dictfile, 'rb'))

  if task == 'tagging':
    records = list(map(
        lambda r: [list(map(lambda w: dicts['w2i'][w], r[0])),
                   list(map(lambda t: dicts['t2i'][t], r[1]))]
        records
    ))
  elif task == 'classification':
    records = list(map(
        lambda r: [list(map(lambda w: dicts['w2i'][w], r[0])),
                   dicts['t2i'][t]]
        records
    ))

  pkl.dump({'meta': meta, 'records': records},
           open(datafile, 'wb'))
