import pandas as pd
import pickle
from gensim import matutils, models
import scipy.sparse
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_pickle('dtm_stop.pkl')
tdm = data.transpose()

sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)

cv = pickle.load(open('cv_stop.pkl', 'rb'))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())

# lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, passes=10)
# print(lda.print_topics())


def nouns(txt):
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(txt)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
    return ' '.join(all_nouns)


def nouns_adj(txt):
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(txt)
    nouns_adjectives = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)]
    return ' '.join(nouns_adjectives)


data_clean = pd.read_pickle('data_clean.pkl')
data_nouns = pd.DataFrame(data_clean.post.apply(nouns))

additional_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                         'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said', 'georgia', 'tech']
stop_words = text.ENGLISH_STOP_WORDS.union(additional_stop_words)

cvn = CountVectorizer(stop_words=stop_words)
data_cvn = cvn.fit_transform(data_nouns.post)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.index

corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())

ldan = models.LdaModel(corpus=corpusn, num_topics=5, id2word=id2wordn, passes=10)
print(ldan.print_topics())

data_nouns_adj = pd.DataFrame(data_clean.post.apply(nouns_adj))

cvna = CountVectorizer(stop_words=stop_words, max_df=0.8)
data_cvna = cvna.fit_transform(data_nouns_adj.post)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index

corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())

ldana = models.LdaModel(corpus=corpusna, num_topics=5, id2word=id2wordna, passes=10)
print(ldana.print_topics())

# 0: campus, 1: vaccine, 2: police, 3: students, 4: mental health
