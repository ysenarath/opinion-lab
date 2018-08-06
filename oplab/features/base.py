import logging
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from tklearn.feature_extraction import TransferFeaturizer, EmbeddingTransformer, LexiconVectorizer
from tklearn.preprocessing import DictionaryTokenizer, TweetTokenizer
from tklearn.text.tokens import build_vocabulary

from oplab.cache import google_word2vec, context2vec, twitter_word2vec, twitter_glove, wiki_fasttext, \
    sentiment140_doc2vec
from oplab.config import DEEP_MOJI_PATH, MODEL_PATH
from oplab.features.deep_moji import DeepMojiTransformer

logging.getLogger(__name__).addHandler(logging.StreamHandler())


class Echo:
    def __call__(self, *args, **kwargs):
        return args[0]


class SplitSelect:
    def __init__(self, delimiter, idx):
        self.delimiter = delimiter
        self.idx = idx

    def __call__(self, texts, *args, **kwargs):
        return map(lambda text: text.split(self.delimiter)[self.idx], texts)


class Doc2VecLookup:
    def __init__(self, doc2vec='sentiment140'):
        self.doc2vec = doc2vec

    def __call__(self, texts, *args, **kwargs):
        if self.doc2vec == 'sentiment140':
            doc2vec = sentiment140_doc2vec()
        else:
            doc2vec = sentiment140_doc2vec()
        return map(doc2vec.infer_vector, texts)


def make_embedding_pipeline(dataset, features):
    if len(features) != 1:
        logging.warning('Embedding features support only if there is one embedding type.')
    pipe = []
    features = [t['name'] for t in features if 'name' in t]
    padding = features[0]['padding'] if len(features) > 0 and 'padding' in features[0] else 50
    word_vectors = None
    if 'google_word2vec' in features:
        word_vectors = google_word2vec()
    elif 'google_word2vec-emoji2vec' in features:
        word_vectors = google_word2vec(emoji2vec=True)
    elif 'twitter_word2vec' in features:
        word_vectors = twitter_word2vec()
    elif 'wiki_fasttext' in features:
        word_vectors = wiki_fasttext()
    elif 'wiki_fasttext-subword' in features:
        word_vectors = wiki_fasttext(subword=True)
    elif 'twitter_glove' in features:
        word_vectors = twitter_glove()
    assert word_vectors is not None, 'No embedding selected. Please provide a valid embedding.'
    if 'right_only' in features:
        sel = FunctionTransformer(SplitSelect('[#TRIGGERWORD#]', 1), validate=False)
        pipe.append(sel)
    elif 'left_only' in features:
        sel = FunctionTransformer(SplitSelect('[#TRIGGERWORD#]', 0), validate=False)
        pipe.append(sel)
    if 'only_dict_tok' in features:
        tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=False)
    elif 'dict_tok' in features:
        tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
    else:
        tt = TweetTokenizer()
    pipe.append(tt)
    vocab = build_vocabulary(*[ds[0] for ds in dataset], tokenizer=tt)
    ee = EmbeddingTransformer(word_vectors, vocab, pad_sequences=padding, output='matrix', default='random')
    pipe.append(ee)
    return pipe


def make_fvector_pipeline(dataset, features):
    pipe = []
    for feature in features:
        feature_name = feature['name']
        if 'google_word2vec' == feature_name:
            word_vectors = google_word2vec()
            tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
            vocab = build_vocabulary(*[ds[0] for ds in dataset], tokenizer=tt)
            ea = EmbeddingTransformer(word_vectors, vocab, output='average', default='ignore')
            pipe.append(make_pipeline(tt, ea))
        elif 'google_word2vec-emoji2vec' == feature_name:
            word_vectors = google_word2vec(emoji2vec=True)
            tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
            vocab = build_vocabulary(*[ds[0] for ds in dataset], tokenizer=tt)
            ea = EmbeddingTransformer(word_vectors, vocab, output='average', default='ignore')
            pipe.append(make_pipeline(tt, ea))
        elif 'twitter_word2vec' == feature_name:
            word_vectors = twitter_word2vec()
            tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
            vocab = build_vocabulary(*[ds[0] for ds in dataset], tokenizer=tt)
            ea = EmbeddingTransformer(word_vectors, vocab, output='average', default='ignore')
            pipe.append(make_pipeline(tt, ea))
        elif 'wiki_fasttext' == feature_name:
            word_vectors = wiki_fasttext()
            tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
            vocab = build_vocabulary(*[ds[0] for ds in dataset], tokenizer=tt)
            ea = EmbeddingTransformer(word_vectors, vocab, output='average', default='ignore')
            pipe.append(make_pipeline(tt, ea))
        elif 'wiki_fasttext-subword' == feature_name:
            word_vectors = wiki_fasttext(subword=True)
            tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
            vocab = build_vocabulary(*[ds[0] for ds in dataset], tokenizer=tt)
            ea = EmbeddingTransformer(word_vectors, vocab, output='average', default='ignore')
            pipe.append(make_pipeline(tt, ea))
        elif 'twitter_glove' == feature_name:
            word_vectors = twitter_glove()
            tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
            vocab = build_vocabulary(*[ds[0] for ds in dataset], tokenizer=tt)
            ea = EmbeddingTransformer(word_vectors, vocab, output='vector', default='ignore')
            pipe.append(make_pipeline(tt, ea))
        elif 'context2vec' == feature_name:
            c2v = FunctionTransformer(context2vec, validate=False)
            pipe.append(c2v)
        elif 'unigram' == feature_name:
            pipe.append(make_pipeline(
                TweetTokenizer(),
                CountVectorizer(preprocessor=Echo(), tokenizer=Echo())
            ))
        elif 'bigram' == feature_name:
            pipe.append(make_pipeline(
                TweetTokenizer(),
                CountVectorizer(preprocessor=Echo(), tokenizer=Echo(), ngram_range=(2, 2))
            ))
        elif 'tf-idf' == feature_name:
            pipe.append(make_pipeline(
                TweetTokenizer(),
                TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
            ))
        elif 'deep_moji' == feature_name:
            if os.path.exists(DEEP_MOJI_PATH):
                ec = DeepMojiTransformer(DEEP_MOJI_PATH)
                pipe.append(ec)
            else:
                logging.getLogger(__name__) \
                    .warning('Can\'t find model named \'deep_moji\' in model folder at {}.'
                             'Ignoring \'emo_clf\' feature.'.format(DEEP_MOJI_PATH))
        elif 'deep_moji_att' == feature_name:
            if os.path.exists(DEEP_MOJI_PATH):
                ec = DeepMojiTransformer(DEEP_MOJI_PATH, return_attention=True)
                pipe.append(ec)
            else:
                logging.getLogger(__name__) \
                    .warning('Can\'t find model named \'deep_moji\' in model folder at {}.'
                             'Ignoring \'emo_clf\' feature.'.format(DEEP_MOJI_PATH))
        elif LexiconVectorizer.has_filter(feature_name):
            atv = LexiconVectorizer(feature_name, True)
            pipe.append(atv)
        elif 'doc2vec' in features:
            pipe.append(make_pipeline(
                TweetTokenizer(),
                FunctionTransformer(Doc2VecLookup(), validate=False)
            ))
        else:
            model_file = os.path.join(MODEL_PATH, feature_name)
            if os.path.exists(model_file):
                ec = TransferFeaturizer(model_file)
                pipe.append(ec)
            else:
                logging.getLogger(__name__) \
                    .warning('Unable to use feature named {}. Ignoring and continuing...'.format(feature_name))
    assert len(pipe) != 0, 'No features selected. Please provide at least one valid feature.'
    return make_union(*pipe)
