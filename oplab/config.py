import os

#  Path to the global resource
ENV_RESOURCE = 'D:\Documents\Resources\\tkresources'

# Dataset path
DATASETS = {
    'emoint': os.path.join(ENV_RESOURCE, 'datasets', 'EmoInt-2017'),
    'aitec': os.path.join(ENV_RESOURCE, 'datasets', 'SemEval-2018'),
    'iest': os.path.join(ENV_RESOURCE, 'datasets', 'IEST-2018'),
    'sentiment140': os.path.join(ENV_RESOURCE, 'datasets', 'Sentiment140', 'training.1600000.processed.noemoticon.csv'),
}

GLOVE_PATH = os.path.join(ENV_RESOURCE, 'models', 'glove.twitter.27B', 'glove.twitter.27B.200d.txt')
GOOGLE_E2V_PATH = os.path.join(ENV_RESOURCE, 'models', 'emoji2vec', 'emoji2vec.txt.gz')
GOOGLE_W2V_PATH = os.path.join(ENV_RESOURCE, 'models', 'google.word2vec', 'GoogleNews-vectors-negative300.bin')
TWITTER_W2V_PATH = os.path.join(ENV_RESOURCE, 'models', 'twitter.word2vec', 'word2vec_twitter_model.bin')
WIKIS_FT_PATH = os.path.join(ENV_RESOURCE, 'models', 'fastText', 'wiki-news-300d-1M-subword.vec.gz')
WIKI_FT_PATH = os.path.join(ENV_RESOURCE, 'models', 'fastText', 'wiki-news-300d-1M.vec.gz')
DEEP_MOJI_PATH = os.path.join(ENV_RESOURCE, 'models', 'DeepMoji')
DOC2VEC_PATH = ''

#  Place where the models are stored
PROJECT_PATH = 'D:\Documents\Workspace\Academic\Masters\DataSEARCH\Project\\opinion-lab'
MODEL_PATH = '{}\models'.format(PROJECT_PATH)
TEMP_PATH = '{}\\temp'.format(PROJECT_PATH)
