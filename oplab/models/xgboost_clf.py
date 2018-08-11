import pprint
import sys

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from tklearn.model_selection import HyperoptOptimizer
from xgboost import XGBClassifier

from oplab.features import make_fvector_pipeline


def xgb_clf(max_depth=7, min_child_weight=1, learning_rate=0.1, n_estimators=500, silent=True,
            objective='binary:logistic', gamma=0, max_delta_step=0, subsample=1, colsample_bytree=1,
            colsample_bylevel=1, reg_alpha=0, reg_lambda=0, scale_pos_weight=1, seed=1, missing=None, **kwargs):
    return XGBClassifier(max_depth=max_depth, min_child_weight=min_child_weight, learning_rate=learning_rate,
                         n_estimators=n_estimators, silent=silent, objective=objective, gamma=gamma,
                         max_delta_step=max_delta_step, subsample=subsample, colsample_bytree=colsample_bytree,
                         colsample_bylevel=colsample_bylevel, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                         scale_pos_weight=scale_pos_weight, seed=seed, missing=missing)


def print_info(params, loss, X_test, y_test, y_pred, **kwargs):
    info = {'params': params, 'loss': loss}
    info_pretty = pprint.pformat(info, indent=4)
    print(info_pretty, file=sys.stderr, flush=True)


def build_model(dataset, features=None, model=None, **kwargs):
    if features is None:
        features = []
    if model is None:
        model = {}
    embedding_extractor = make_fvector_pipeline(dataset, features)
    clf = xgb_clf(**model)
    return make_pipeline(embedding_extractor, clf).fit(dataset[0][0], dataset[0][1])


def optimize_model(dataset, features=None, model_cfg=None, scorer=None, n_samples=None, max_evals=25):
    if features is None:
        features = []
    if model_cfg is None:
        model_cfg = {}
    if scorer is None:
        def scorer(**kwargs):
            return 0.0
    if n_samples is None:
        n_samples = len(dataset[0][0])
        n_samples = 48000 if n_samples >= 48000 else n_samples
    ds_text, ds_labels = resample(dataset[0][0], dataset[0][1], n_samples=n_samples, random_state=0)
    text_train, text_test, y_train, y_test = train_test_split(ds_text, ds_labels, test_size=0.4)
    feature_extractor = make_fvector_pipeline(dataset, features)
    X_train = feature_extractor.fit_transform(text_train, y_train)
    X_test = feature_extractor.transform(text_test)
    clf = XGBClassifier
    opt = HyperoptOptimizer(clf, model_cfg, scorer, max_evals, callbacks=[print_info]).optimize(X_train, X_test,
                                                                                                y_train, y_test)
    return opt
