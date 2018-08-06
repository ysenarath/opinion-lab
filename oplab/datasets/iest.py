import numpy as np
from nltk import tokenize
from sklearn.preprocessing import LabelEncoder
from tklearn.datasets import load_iest

from oplab.config import DATASETS

def preprocess(texts, y, filters=None):
    """
    Dataset level pre-processing
    :param texts:
    :param y:
    :param filters:
    :return:
    """
    if filters is None:
        filters = []
    for fx in filters:
        try:
            texts, y = list(zip(*[(tx, ty) for (tx, ty) in zip(texts, y) if fx(tx)]))
        except:
            pass
    dx = list(zip(
        *[(tx.replace('[#TRIGGERWORD#]', '__trigger__').replace('[NEWLINE]', ' __newline__ '), ty) for (tx, ty) in
          zip(texts, y) if '[#TRIGGERWORD#]' in tx]))
    return dx


def clean_text(texts, *args):
    return list(map(lambda x: x.replace('[#TRIGGERWORD#]', '__trigger__').replace('[NEWLINE]', ' __newline__ '), texts))


def read_dataset(filters=None, include_test=False, **kwargs):
    train_data, trial_data, test_data = load_iest(DATASETS['iest'])

    train_labels, train_text = list(zip(*train_data))
    trial_labels, trial_text = list(zip(*trial_data))
    test_labels, test_text = list(zip(*test_data))

    enc = LabelEncoder()
    train_y = enc.fit_transform(train_labels)
    trial_y = enc.transform(trial_labels)
    test_y = enc.transform(test_labels)

    train_text, train_y = preprocess(train_text, train_y, filters=filters)
    trial_text = clean_text(trial_text)
    test_text = clean_text(test_text)

    #  Remove sentences without [#TRIGGERWORD#]
    if include_test:
        train_text = list(train_text) + trial_text
        train_y = np.array(list(train_y) + trial_y.tolist())
        output = [(train_text, train_y), (test_text, test_y)]
    else:
        output = [(train_text, train_y), (trial_text, trial_y)]

    return output, enc.classes_


if __name__ == '__main__':
    c1 = read_dataset(include_test=True)[0][1]
    c2 = read_dataset()[0][1]
    assert len(c1) > len(c2), 'Train datasets is larger than test datasets.'
