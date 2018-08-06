import json
import os

from deepmoji.model_def import deepmoji_feature_encoding
from deepmoji.sentence_tokenizer import SentenceTokenizer
from sklearn.base import TransformerMixin, BaseEstimator


class DeepMojiTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, model_path, return_attention=False, max_len=10):
        self.model_path = model_path
        self.return_attention = return_attention
        self.max_len = max_len
        self.initialize()

    def initialize(self):
        deepmoji_weights_path = os.path.join(self.model_path, 'deepmoji_weights.hdf5')
        vocabulary_path = os.path.join(self.model_path, 'vocabulary.json')
        with open(vocabulary_path, 'r') as f:
            vocab = json.load(f)
        self._st_ = SentenceTokenizer(vocab, self.max_len)
        self._model_ = deepmoji_feature_encoding(self.max_len, deepmoji_weights_path, self.return_attention)

    def fit(self, X, *_):
        return self

    def transform(self, X, *_):
        tokens, _, _ = self._st_.tokenize_sentences(X)
        vecs = self._model_.predict(tokens)
        if self.return_attention:
            return vecs[1]
        return vecs

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def __setstate__(self, state):
        self.model_path = state['model_path']
        self.max_len = state['max_len']
        self.return_attention = state['return_attention']
        self.initialize()

    def __getstate__(self):
        return {
            'model_path': self.model_path,
            'max_len': self.max_len,
            'return_attention': self.return_attention,
        }
