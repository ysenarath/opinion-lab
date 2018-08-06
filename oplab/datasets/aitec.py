from tklearn.datasets.load_ait import load_ait

from oplab.config import DATASETS


def read_dataset(**kwargs):
    train, dev, test = load_ait(DATASETS['aitec'], 'E.c')

    categories = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
                  'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
    train_y = train[categories].values
    train_text = train['Tweet'].values
    trial_y = dev[categories].values
    trial_text = dev['Tweet'].values

    return [(train_text, train_y), (trial_text, trial_y)], None
