import os
import re
import time

from gensim.models import Doc2Vec
from tklearn.text.embedding import load_word2vec, load_glove, load_embedding
from tklearn.utils import DiskCache, WebService

from oplab.config import DOC2VEC_PATH
from oplab.config import GLOVE_PATH
from oplab.config import GOOGLE_E2V_PATH
from oplab.config import GOOGLE_W2V_PATH
from oplab.config import TEMP_PATH
from oplab.config import TWITTER_W2V_PATH
from oplab.config import WIKIS_FT_PATH
from oplab.config import WIKI_FT_PATH

context2vec_cache = None  # Dick Cache
google_word2vec_cache = None  # Mem Cache
twitter_word2vec_cache = None  # Mem Cache
twitter_glove_cache = None  # Mem Cache
google_emoji2vec_cache = None  # Mem Cache
wiki_fasttext_cache = None  # Mem Cache
sentiment140_doc2vec_cache = None  # Mem Cache


def twitter_glove(verbose=False):
    """

    :return:
    """
    global twitter_glove_cache
    if twitter_glove_cache is not None:
        return twitter_glove_cache
    # Load word2vec
    if verbose:
        print('Loading Glove...', end=' ', flush=True)
    start = time.time()
    word_vec = load_glove(GLOVE_PATH)
    end = time.time()
    if verbose:
        print('\t[{} seconds]'.format(end - start))
    twitter_glove_cache = word_vec
    return twitter_glove_cache


def google_word2vec(emoji2vec=False, verbose=False):
    """

    :return:
    """
    global google_word2vec_cache
    if google_word2vec_cache is not None:
        return google_word2vec_cache
    # Load word2vec
    if verbose:
        print('Loading Word2vec...', end=' ', flush=True)
    start = time.time()
    word_vec = load_word2vec(GOOGLE_W2V_PATH)
    end = time.time()
    if emoji2vec:
        e2v = google_emoji2vec(verbose)
        word_vec.append(e2v)
    if verbose:
        print('\t[{} seconds]'.format(end - start))
    google_word2vec_cache = word_vec
    return google_word2vec_cache


def twitter_word2vec(verbose=False):
    """

    :return:
    """
    global twitter_word2vec_cache
    if twitter_word2vec_cache is not None:
        return twitter_word2vec_cache
    # Load word2vec
    if verbose:
        print('Loading Word2vec...', end=' ', flush=True)
    start = time.time()
    word_vec = load_word2vec(TWITTER_W2V_PATH, binary=True, unicode_errors='ignore')
    end = time.time()
    if verbose:
        print('\t[{} seconds]'.format(end - start))
    twitter_word2vec_cache = word_vec
    return twitter_word2vec_cache


def google_emoji2vec(verbose=False):
    """
    Loads an extension to word2vec trained on google news corpus containing word vectors for emojis
    :return:
    """
    global google_emoji2vec_cache
    if google_emoji2vec_cache is not None:
        return google_emoji2vec_cache
    # Load word2vec
    if verbose:
        print('Loading Emoji2vec...', end=' ', flush=True)
    start = time.time()
    word_vec = load_embedding(GOOGLE_E2V_PATH)
    end = time.time()
    if verbose:
        print('\t[{} seconds]'.format(end - start))
    google_emoji2vec_cache = word_vec
    return google_emoji2vec_cache


def wiki_fasttext(subword=False, verbose=False):
    """

    :return:
    """
    global wiki_fasttext_cache
    if wiki_fasttext_cache is not None:
        return wiki_fasttext_cache
    # Load word2vec
    if verbose:
        print('Loading FastText...', end=' ', flush=True)
    start = time.time()
    if subword:
        word_vec = load_embedding(WIKIS_FT_PATH, word_first=True, leave_head=True)
    else:
        word_vec = load_embedding(WIKI_FT_PATH, word_first=True, leave_head=True)
    end = time.time()
    if verbose:
        print('\t[{} seconds]'.format(end - start))
    wiki_fasttext_cache = word_vec
    return wiki_fasttext_cache


def context2vec(tweets, verbose=False):
    """
    Extract context2vec Embeddings for given list of tweets. Target Word ([#TRIGGERWORD#])
    considered as the word tobe predicted.

    :param tweets: A collection of tweets
    :return: extracted context2vec features.
    """
    global context2vec_cache
    if context2vec_cache is None:
        context2vec_cache = DiskCache(os.path.join(TEMP_PATH, 'context2vec.cache'))
    ws = WebService('127.0.0.1', 5000)
    result = []
    target_re = '\s?__trigger__\s?'
    if verbose:
        print('Extracting context2vec vectors...')
    for i, tweet in enumerate(tweets):
        progress = int((i + 1) * 30 / len(tweets))
        if verbose:
            print('\rProgress: [' + '=' * progress + '.' * (30 - progress) + ']', end='', flush=True)
        text = re.sub(target_re, ' [] ', tweet)
        if context2vec_cache.has(text):
            result += [context2vec_cache[text]]
        else:
            c2v = ws.context2vec(text=text)
            context2vec_cache[text] = c2v
            result += [c2v]
    if verbose:
        print()  # End progress line
    context2vec_cache.flush()
    return result


def sentiment140_doc2vec(verbose=False):
    """

    :return:
    """
    global sentiment140_doc2vec_cache
    if sentiment140_doc2vec_cache is not None:
        return sentiment140_doc2vec_cache
    # Load Doc2Vec
    if verbose:
        print('Loading Word2vec...', end=' ', flush=True)
    start = time.time()
    doc2vec = Doc2Vec.load(DOC2VEC_PATH)
    end = time.time()
    if verbose:
        print('\t[{} seconds]'.format(end - start))
    sentiment140_doc2vec_cache = doc2vec
    return sentiment140_doc2vec_cache
