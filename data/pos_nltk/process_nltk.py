import nltk, numpy as np
from sklearn.model_selection import train_test_split

def split_and_write_to_file(x, y, filename):
    with open(filename, 'w') as f:
        for _x, _y in zip(x, y):
            f.write(' '.join(_x) + "|||" + ' '.join(_y) + "\n")
dataset = nltk.corpus.treebank.tagged_sents()

s, t = [], []

for data in dataset:
    _s, _t = zip(*data)
    s.append(np.array([w.lower() for w in _s]))
    t.append(np.array(_t))

x_train, x_test, y_train, y_test = train_test_split(s, t, test_size=0.2)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

split_and_write_to_file(x_train, y_train, 'train.txt')
split_and_write_to_file(x_val, y_val, 'dev.txt')
split_and_write_to_file(x_test, y_test, 'test.txt')
