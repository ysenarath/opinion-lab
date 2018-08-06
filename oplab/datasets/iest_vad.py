import numpy as np
from tklearn.datasets import load_iest

from oplab.config import DATASETS


def to_vad(t):
    #  ['anger' 'disgust' 'fear' 'joy' 'sad' 'surprise']
    if t == 'sad':
        return 0, 0, 0
    elif t == 'anger':
        return 0, 1, 1
    elif t == 'surprise':
        return 1, 1, 0
    elif t == 'fear':
        return 0, 1, 0
    elif t == 'disgust':
        return 0, 1, 1
    elif t == 'joy':
        return 1, 1, 1
    else:
        return 0, 0, 0


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


def read_dataset(filters=None, **kwargs):
    train_data, trial_data, _ = load_iest(DATASETS['iest'])

    train_labels, train_text = list(zip(*train_data))
    trial_labels, trial_text = list(zip(*trial_data))

    train_y = map(to_vad, train_labels)
    trial_y = map(to_vad, trial_labels)

    train_text, train_y = preprocess(train_text, train_y, filters=filters)
    trial_text, trial_y = preprocess(trial_text, trial_y)

    #  Remove sentences without [#TRIGGERWORD#]
    output = [(train_text, np.array(train_y)), (trial_text, np.array(trial_y))]

    return output, ['pleasure', 'arousal', 'dominance']


if __name__ == '__main__':
    out, c = read_dataset(DATASETS['iest'], return_categories=True)
