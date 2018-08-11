from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.utils import resample
from tklearn.model_selection import HyperoptOptimizer

from oplab.features import make_fvector_pipeline


def build_model(dataset, features, model, **kwargs):
    feature_extractor = make_fvector_pipeline(dataset, features)
    model_params = {}
    if 'C' in model:
        model_params['C'] = model['C']
    clf = SVC(**model_params)
    return make_pipeline(feature_extractor, clf).fit(dataset[0][0], dataset[0][1])


def optimize_model(dataset, features, model_cfg=None, scorer=None, n_samples=None, max_evals=10):
    if model_cfg is None:
        model_cfg = {}
    if scorer is None:
        def scorer(**kwargs):
            return 0.0
    if n_samples is None:
        n_samples = len(dataset[0][0])
        n_samples = 6000 if n_samples >= 6000 else n_samples
    ds_text, ds_labels = resample(dataset[0][0], dataset[0][1], n_samples=n_samples, random_state=0)
    text_train, text_test, y_train, y_test = train_test_split(ds_text, ds_labels, test_size=0.4)
    feature_extractor = make_fvector_pipeline(dataset, features)
    X_train = feature_extractor.fit_transform(text_train, y_train)
    X_test = feature_extractor.transform(text_test)
    clf = SVC
    opt = HyperoptOptimizer(clf, model_cfg, scorer, max_evals).optimize(X_train, X_test, y_train, y_test)
    return opt
