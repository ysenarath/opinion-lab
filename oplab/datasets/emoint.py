import pandas as pd
from tklearn.datasets import load_emoint

from oplab.config import DATASETS

_emoint_emotions = {
    0: 'anger',
    1: 'joy',
    2: 'sadness',
    3: 'fear',
    '0': 'anger',
    '1': 'joy',
    '2': 'sadness',
    '3': 'fear',
    'anger': 'anger',
    'joy': 'joy',
    'sadness': 'sadness',
    'fear': 'fear',
}


def read_dataset(emotion=0, include_test=False, **kwargs):
    train, dev, test = load_emoint(input_path=DATASETS['emoint'])
    emotion = _emoint_emotions[emotion] if emotion in _emoint_emotions else _emoint_emotions[0]
    # Merging datasets
    # -- Training Dataset
    if include_test:
        frames = [train, dev]
        train_dev = pd.concat(frames)
        train_emotion = train_dev[train_dev.emotion == emotion]
        test_emotion = test[test.emotion == emotion]
    else:
        train_emotion = train[train.emotion == emotion]
        test_emotion = dev[dev.emotion == emotion]
    train_x, train_y = train_emotion['text'], train_emotion['rating']
    test_x, test_y = test_emotion['text'], test_emotion['rating']
    return [(train_x, train_y), (test_x, test_y)], None


if __name__ == '__main__':
    dataset = read_dataset(DATASETS['emoint'], emotion=1, include_test=True)
    assert len(dataset[0][0]) == len(dataset[0][1]), 'Invalid training set.'
    assert len(dataset[1][0]) == len(dataset[1][1]), 'Invalid test set.'
    dataset = read_dataset(DATASETS['emoint'], emotion=1)
    assert len(dataset[1][0]) == len(dataset[1][1]), 'Invalid test set.'
    assert len(dataset[1][0]) == len(dataset[1][1]), 'Invalid test set.'
