import pprint
import sys

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from tklearn.model_selection import HyperoptOptimizer
from tklearn.neural_network import FNNClassifier

from oplab.features import make_fvector_pipeline


def print_info(params, loss, X_test, y_test, y_pred, **kwargs):
    info = {'params': params, 'loss': loss}
    info_pretty = pprint.pformat(info, indent=4)
    print(info_pretty, file=sys.stderr, flush=True)


def build_model(dataset, features=None, model=None, **kwargs):
    if features is None:
        features = []
    if model is None:
        model = {}
    feature_extractor = make_fvector_pipeline(dataset, features)
    clf = FNNClassifier(**model)
    if 'early_stopping' in model:
        clf.early_stopping(**model)
    if 'validation_split' in model:
        clf.validation_split(**model)
    pipe = make_pipeline(feature_extractor, clf).fit(dataset[0][0], dataset[0][1])
    dash_line = '\n' + '-' * 50 + '\n'
    print('{}Training Completed.\nFeature Size: {}{}'.format(dash_line, clf.num_features_, dash_line))
    return pipe


def optimize_model(dataset, features=None, model_cfg=None, scorer=None, n_samples=None, max_evals=15):
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
    embedding_extractor = make_fvector_pipeline(dataset, features)
    X_train = embedding_extractor.fit_transform(text_train, y_train)
    X_test = embedding_extractor.transform(text_test)
    opt = HyperoptOptimizer(FNNClassifier, model_cfg, scorer, max_evals, callbacks=[print_info]).optimize(X_train,
                                                                                                          X_test,
                                                                                                          y_train,
                                                                                                          y_test)
    return opt
