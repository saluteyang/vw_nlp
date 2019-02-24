import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk, re, pprint, string
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob, Word
from wordcloud import WordCloud

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

from gensim import corpora, models, similarities, matutils

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from helpers import ENGLISH_STOP_WORDS, chunks, display_topics, \
    potential_stop_words, plotSentiment


stop_words = stopwords.words('english')
stop_words_v0 = stop_words + ["man", "men", "woman", "women", "mr", "mrs", "ramsay", "chapter"]
stop_words_v1 = stop_words_v0 + ["macalister", "charles", "tansley", "minta", "paul", "rayley",
                                  "beckwith", "william", "bankes", "carmichael", "mcnab", "husband",
                                  "wife", "son", "girl", "nancy", "joseph", "mildred", "scott",
                                  "augustus", "carrie", "mary", "damn"]
stop_words_v2 = stop_words_v1 + ["felt", "thought", "wanted", "knew", "looked", "looking", "liked", "say", "came"]
stop_words_v3 = stop_words_v2 + ["look", "come", "went", "saw", "things", "people", "time", "mind",
                                  "way", "old", "thinking", "going", "lily", "james", "eyes", "cam",
                                  "moment", "children", "thing", "room", "table", "let", "house", "hand",
                                  "away", "father"]
stop_words_v4 = stop_words_v3 + ["something", "little", "nothing", "never", "always", "suddenly", "still",
                                 "even", "perhaps", "anything", "good", "well", "life", "light", "world",
                                 "book", "day", "long", "word", "round", "great", "light", "together", "alone"]
stop_words_v5 = stop_words_v4 + ["back", "year", "head", "tree", "whole", "rather", "quite", "half",
                                 "window", "last", "much", "ever", "everything", "feeling"]

# use different document segmentation
# use pretrained sentiment model for the chapters of To the Lighthouse
# use on a different text Mrs Dalloway?
# sentiment across novels

# with open('txt_data/orlando_1928.txt') as f:
#     text = f.readlines()

with open('txt_data/to_the_lighthouse_1927.txt') as f:
    text = f.readlines()


text = [''.join(text)][0]
text = TextBlob(text).words

# remove digits
text_clean = [re.sub('\w*\d\w*', ' ', x) for x in text]
# remove punctuation and lower case all words
text_clean = [re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower()) for x in text_clean]
# remove newline character
text_clean = [x.replace('\n', '') for x in text_clean]

# stemming
# st = PorterStemmer()
# text_clean = [st.stem(x) for x in text_clean]
# lemmatizing
# text_clean = [Word(x).lemmatize() for x in text_clean]

# vectorizer = CountVectorizer(stop_words=potential_stop_words,
#                              ngram_range=(1, 1),
#                              max_df=0.8, min_df=2)
# doc_word = vectorizer.fit_transform(docs)
# doc_word.shape

#############################################################
# re-forming words in doc into docs
docs = [(' ').join(x) for x in chunks(text_clean, 50)]

# custom stop words
vectorizer = TfidfVectorizer(stop_words=potential_stop_words2,
                             ngram_range=(1, 1),
                             max_df=0.9)
doc_word = vectorizer.fit_transform(docs)
doc_word.shape

# np.savetxt('50_doc_word_out', doc_word.toarray(),  delimiter='\t')

# NMF model
nmf_model = NMF(3)
doc_topic = nmf_model.fit_transform(doc_word)
display_topics(nmf_model, vectorizer.get_feature_names(), 10)

# np.savetxt('50_doc_out', doc_topic, delimiter='\t')

# NMF model 2
docs = [(' ').join(x) for x in chunks(text_clean, 5)]

# custom stop words
vectorizer = TfidfVectorizer(stop_words=potential_stop_words2,
                             ngram_range=(1, 1),
                             max_df=0.8)

doc_word = vectorizer.fit_transform(docs)
doc_word.shape

nmf_model2 = NMF(4)
doc_topic2 = nmf_model2.fit_transform(doc_word)
display_topics(nmf_model2, vectorizer.get_feature_names(), 10)

# sentiment analysis
docs = [(' ').join(x) for x in chunks(text_clean, 10)]
text_sent = [TextBlob(x).sentiment for x in docs]
text_polar = []
for i in range(len(text_sent)):
    text_polar.append(text_sent[i][0])

# plt.plot(text_polar, marker='o')
# plt.show()

plotSentiment(text_polar, 10, save=True, savefile='text_sent_10.png')

# split based on chapter numbering
with open('txt_data/to_the_lighthouse_1927.txt') as f:
    text = f.readlines()
text = [''.join(text)][0]

# split based on digits
text_chap = re.compile('\w*\d\w*').split(text)[1:]  # remove first section header

text_sent_ch = [TextBlob(x).sentiment for x in text_chap]
text_polar_ch = []
for i in range(len(text_sent_ch)):
    text_polar_ch.append(text_sent_ch[i][0])

plotSentiment(text_polar_ch, len(text_chap), save=True, savefile='text_sent_ch.png')

from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label, LabelSet, ColumnDataSource
from bokeh.io import output_notebook
import seaborn as sns
import matplotlib.colors as mcolors

# Dominant topic number in each doc
topic_num = np.argmax(doc_topic2, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_nmf = tsne_model.fit_transform(doc_topic2)

# Plot the Topic Clusters using Bokeh
TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save"
n_topics = 4
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

# create a df for tooltip assignment
doc_topic_p = pd.DataFrame({'topic': topic_num,
                            'topic_idx': range(1, doc_topic2.shape[0] + 1),
                            'x': tsne_nmf[:, 0],
                            'y': tsne_nmf[:, 1],
                            'color': mycolors[topic_num]})

plot = figure(title="t-SNE Clustering of {} NMF Topics".format(n_topics),
              plot_width=900, plot_height=700, tools=TOOLS)
plot.hover.tooltips = [
    ("document topic", "@topic"),
    ("document number", "@topic_idx")
]
source = ColumnDataSource(doc_topic_p)
plot.scatter(x='x', y='y', color='color', source=source, size=10)
show(plot)

# try NMF on chapter numbering
# split based on chapter numbering
with open('txt_data/to_the_lighthouse_1927.txt') as f:
    text = f.readlines()
text = [''.join(text)][0]

# split based on digits
text_chap = re.compile('\w*\d\w*').split(text)[1:]  # remove first section header
text_chap = [[word for word in simple_preprocess(str(doc))] for doc in text_chap]
docs = [' '.join(x) for x in text_chap]

vectorizer = TfidfVectorizer(stop_words=potential_stop_words2,
                             ngram_range=(1, 1),
                             max_df=0.9)
doc_word = vectorizer.fit_transform(docs)
doc_word.shape

# NMF model
nmf_model = NMF(3)
doc_topic = nmf_model.fit_transform(doc_word)
display_topics(nmf_model, vectorizer.get_feature_names(), 10)


# remove sentiment outliers
# text_polar_chx = text_polar_ch
# text_polar_chx[1] = (text_polar_ch[0] + text_polar_ch[2])/2
# text_polar_chx[38] = (text_polar_ch[37] + text_polar_ch[39])/2
# plotSentiment(text_polar_chx, len(text_chap))

# topic results (5 documents, 4 topics, 0.8 max_df)
# Topic  0
# brush, grass, green, summer, silence, sent, canvas, boots, shore, wall
# Topic  1
# love, watch, course, dish, true, listening, french, sort, happened, plate
# Topic  2
# blundered, severity, urn, blame, odious, circus, tobacco, hope, leg, stopping
# Topic  3
# lights, lost, flounder, grow, love, reason, necklace, bit, oneself, brooch

#######
# topic 0: surroundings, environment
# topic 1: garbage
# topic 2: male characters
# topic 3: female characters
#######

# topic results (50 documents, 3 topics, 0.9 max_df)
# Topic  0
# lily, rose, people, life, saying, things, room, saw, love, eyes
# Topic  1
# cam, boat, james, father, lighthouse, sea, old, look, book, sat
# Topic  2
# house, night, summer, light, glass, bast, room, wall, left, long

# topic results (43 documents, 3 topics, 0.9 max_df, by chapter)
# Topic  0
# said, did, lily, little, like, things, people, andrew, picture, eyes
# Topic  1
# james, cam, father, boat, like, lighthouse, sea, said, book, look
# Topic  2
# glass, night, house, room, light, summer, airs, bedroom, long, wall

#############################################################
# SVD and truncate and show variance explained
# lsa = TruncatedSVD(5)
# doc_topic = lsa.fit_transform(doc_word)
# print(sum(lsa.explained_variance_ratio_))
# display_topics(lsa, vectorizer.get_feature_names(), 10)

import gensim, spacy
from gensim.utils import lemmatize, simple_preprocess
import gensim.corpora as corpora

# temp = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in docs]

def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    # texts = [bigram_mod[doc] for doc in texts]
    # texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
    return texts_out

data_ready = process_words(docs, stop_words_v5)

# Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=3,
                                           random_state=100,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

lda_model.print_topics()

#######################################################
# LDA model (need count vectorizer, not tf-idf)
doc_word = doc_word.transpose()
# pd.DataFrame(doc_word.toarray(), vectorizer.get_feature_names()).head()

# Convert sparse matrix of counts to a gensim corpus
corpus = matutils.Sparse2Corpus(doc_word)
id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())

lda = models.LdaModel(corpus=corpus, num_topics=3, id2word=id2word, passes=5)
lda.print_topics()

# try clustering on noun phrases
text_blob = TextBlob(' '.join(text_clean)).noun_phrases

# exclude noun_phrases that include stop words
text_blob_new = []
for item in text_blob:
    if all(subitem not in potential_stop_words for subitem in item.split()):
        text_blob_new.append(item)

# text_blob = [x.replace(' ', '_') for x in text_blob]
# docs_blob = [(' ').join(x) for x in chunks(text_blob, 5)]

text_blob_new = [x.replace(' ', '_') for x in text_blob_new]
docs_blob_new = [(' ').join(x) for x in chunks(text_blob_new, 5)]

vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
# vectorizer = CountVectorizer(stop_words='english')
doc_word_blob = vectorizer.fit_transform(docs_blob_new)
doc_word_blob.shape

vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
doc_word_blob = vectorizer.fit_transform(docs_blob_new)
doc_word_blob.shape

# SVD and truncate and show variance explained
lsa = TruncatedSVD(5)
blob_topic = lsa.fit_transform(doc_word_blob)
print(sum(lsa.explained_variance_ratio_))
display_topics(lsa, vectorizer.get_feature_names(), 10)

# NMF model
nmf_model = NMF(5)
doc_topic = nmf_model.fit_transform(doc_word_blob)
display_topics(nmf_model, vectorizer.get_feature_names(), 10)


# text frequencies
# freq_dict = nltk.FreqDist(text_clean_words)
# sorted_frequency_dict =sorted(freq_dict, key=freq_dict.__getitem__, reverse=True)
# sorted_frequency_dict[:30]
##############################
# text frequencies for noun phrases that don't include stop words
text_clean_str = ''.join(text_clean)  # continue from line 21 before tokenization
text_clean_blob = TextBlob(text_clean_str).noun_phrases
text_clean_blob_temp = []
for item in text_clean_blob:
    if all(subitem not in stopwords.words('english') for subitem in item.split()):
        text_clean_blob_temp.append(item)

# text_clean_blob = [x for x in text_clean_blob if len(x) > 3]
freq_dict2 = nltk.FreqDist(text_clean_blob_temp)
freq_dict2.plot(20, cumulative=False)

# further remove proper names
text_clean_blob_fin = []
for item in text_clean_blob_temp:
    if not(re.match('mr', item) or re.match('mrs', item) or re.match('sir', item)):
        text_clean_blob_fin.append(item)

text_clean_blob_fin = []
for item in text_clean_blob_temp:
    if len(re.findall(r"\bmr\b|\bmrs\b|\bsir\b|\bmacalister\b|\bcharles\b|\bminta\b|"
                      r"\bpaul\b|\bbeckwith\b|\bwilliam\b|\bcarmichael\b|\bman\b|\bmen\b|\bwoman\b|\bwomen\b", item)) == 0:
        text_clean_blob_fin.append(item)

freq_dict3 = nltk.FreqDist(text_clean_blob_fin)
freq_dict3.plot(20, cumulative=False)

# plot frequent nouns using word cloud
wc = WordCloud(width=1600, height=800, max_words=30).generate_from_frequencies(freq_dict3)
plt.figure(figsize=(20, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('orlando_wc.png', facecolor='k', bbox_inches="tight")
plt.show()

