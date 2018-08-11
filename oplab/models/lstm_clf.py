from sklearn.pipeline import make_pipeline
from tklearn.neural_network import LSTMClassifier

from oplab.features import make_embedding_pipeline


def build_model(dataset, features=None, padding=50, model_cfg=None, **kwargs):
    if features is None:
        features = []
    if model_cfg is None:
        model_cfg = {}
    embedding_extractor = make_embedding_pipeline(dataset, padding, features)
    clf = LSTMClassifier(**model_cfg)
    if 'early_stopping' in model_cfg:
        clf.early_stopping(**model_cfg)
    if 'validation_split' in model_cfg:
        clf.validation_split(**model_cfg)
    return make_pipeline(*embedding_extractor, clf).fit(dataset[0][0], dataset[0][1])
