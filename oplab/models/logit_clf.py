from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from tklearn.model_selection import HyperoptOptimizer

from oplab.features import make_fvector_pipeline


def logit_clf(penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
              random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False,
              n_jobs=1, **kwargs):
    return LogisticRegression(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state,
                              solver, max_iter, multi_class, verbose, warm_start, n_jobs)


def build_model(dataset, features=None, model=None, **kwargs):
    if features is None:
        features = []
    if model is None:
        model = {}
    feature_extractor = make_fvector_pipeline(dataset, features)
    clf = logit_clf(**model)
    return make_pipeline(feature_extractor, clf).fit(dataset[0][0], dataset[0][1])


def optimize_model(dataset, features=None, model_cfg=None, scorer=None, n_samples=None, max_evals=10):
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
    feature_extractor = make_fvector_pipeline(dataset, features)
    X_train = feature_extractor.fit_transform(text_train, y_train)
    X_test = feature_extractor.transform(text_test)
    clf = LogisticRegression
    opt = HyperoptOptimizer(clf, model_cfg, scorer, max_evals).optimize(X_train, X_test, y_train, y_test)
    return opt
