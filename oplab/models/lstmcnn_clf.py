import pprint
import sys

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from tklearn.model_selection import HyperoptOptimizer
from tklearn.neural_network import LSTMCNNClassifier

from oplab.features import make_embedding_pipeline


def print_info(params, loss, X_test, y_test, y_pred, **kwargs):
    info = {'params': params, 'loss': loss}
    info_pretty = pprint.pformat(info, indent=4)
    print(info_pretty, file=sys.stderr, flush=True)


def build_model(dataset, features=None, model=None, **kwargs):
    if features is None:
        features = []
    if model is None:
        model = {}
    embedding_extractor = make_embedding_pipeline(dataset, features)
    clf = LSTMCNNClassifier(**model)
    if 'early_stopping' in model:
        clf.early_stopping(**model)
    if 'validation_split' in model:
        clf.validation_split(**model)
    return make_pipeline(*embedding_extractor, clf).fit(dataset[0][0], dataset[0][1])


def optimize_model(dataset, features=None, padding=50, model_cfg=None, scorer=None, n_samples=None, max_evals=10):
    if features is None:
        features = []
    if model_cfg is None:
        model_cfg = {}
    if scorer is None:
        def scorer(**kwargs):
            return 0.0
    if n_samples is None:
        text_train, text_test, y_train, y_test = dataset[0][0], dataset[1][0], dataset[0][1], dataset[1][1]
    else:
        ds_text, ds_labels = resample(dataset[0][0], dataset[0][1], n_samples=n_samples, random_state=0)
        text_train, text_test, y_train, y_test = train_test_split(ds_text, ds_labels, test_size=0.4)
    embedding_extractor = make_embedding_pipeline(dataset, padding, features)
    embedding_extractor = make_pipeline(*embedding_extractor)
    X_train = embedding_extractor.fit_transform(text_train, y_train)
    X_test = embedding_extractor.transform(text_test)
    clf = LSTMCNNClassifier
    opt = HyperoptOptimizer(clf, model_cfg, scorer, max_evals, callbacks=[print_info]).optimize(X_train, X_test,
                                                                                                y_train, y_test)
    return opt
