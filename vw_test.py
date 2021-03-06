import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity

import nltk, re, pprint, string
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob, Word
from wordcloud import WordCloud

from gensim import corpora, models, similarities, matutils

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label, LabelSet, ColumnDataSource
import matplotlib.colors as mcolors

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim, spacy
from gensim.utils import lemmatize, simple_preprocess
import gensim.corpora as corpora

from helpers import *

# stop_words = stopwords.words('english')
# stop_words_v0 = stop_words + ["man", "men", "woman", "women", "mr", "mrs", "ramsay", "chapter"]
# stop_words_v1 = stop_words_v0 + ["macalister", "charles", "tansley", "minta", "paul", "rayley",
#                                   "beckwith", "william", "bankes", "carmichael", "mcnab", "husband",
#                                   "wife", "son", "girl", "nancy", "joseph", "mildred", "scott",
#                                   "augustus", "carrie", "mary", "damn"]
# stop_words_v2 = stop_words_v1 + ["felt", "thought", "wanted", "knew", "looked", "looking", "liked", "say", "came"]
# stop_words_v3 = stop_words_v2 + ["look", "come", "went", "saw", "things", "people", "time", "mind",
#                                   "way", "old", "thinking", "going", "lily", "james", "eyes", "cam",
#                                   "moment", "children", "thing", "room", "table", "let", "house", "hand",
#                                   "away", "father"]
# stop_words_v4 = stop_words_v3 + ["something", "little", "nothing", "never", "always", "suddenly", "still",
#                                  "even", "perhaps", "anything", "good", "well", "life", "light", "world",
#                                  "book", "day", "long", "word", "round", "great", "light", "together", "alone"]
# stop_words_v5 = stop_words_v4 + ["back", "year", "head", "tree", "whole", "rather", "quite", "half",
#                                  "window", "last", "much", "ever", "everything", "feeling"]

with open('txt_data/to_the_lighthouse_1927.txt') as f:
    text = f.readlines()

# with open('txt_data/mrs_dalloway_1925.txt') as f:
#     text = f.readlines()

# with open('txt_data/jacobs_room_1922.txt') as f:
#     text = f.readlines()

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

#############################################################
# re-forming words in doc into docs
docs = [(' ').join(x) for x in chunks(text_clean, 50)]

# docs_concat = docs
# docs_concat = docs_concat + docs
# docs = docs_concat
# with open('docs_all3.txt', 'w') as f:
#     for item in docs:
#         f.write("%s\n" % item)

# custom stop words
vectorizer = TfidfVectorizer(stop_words=potential_stop_words4,
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

# sentiment analysis (10 even chunks)
docs = [(' ').join(x) for x in chunks(text_clean, 10)]
text_sent = [TextBlob(x).sentiment for x in docs]
text_polar = []
for i in range(len(text_sent)):
    text_polar.append(text_sent[i][0])

plotSentiment(text_polar, 10, save=True, savefile='text_sent_10.png')

######################################
# average sentiment scores using TextBlob and Vader
# moving average and sequential average
# TextBlob
plotSentiment(text_polar, 150)
plotSentiment(moving_average(text_polar, n=10), len(text_polar) - 9)

# Vader
analyser = SentimentIntensityAnalyzer()
text_sent2 = [analyser.polarity_scores(x) for x in docs]
text_polar2 = []
for i in range(len(text_sent2)):
    text_polar2.append(text_sent2[i]['compound'])

plotSentiment(moving_average(text_polar2, n=10), len(text_polar) - 9)

# saving side by side comparison
# every 5 point average (50 chunks per novel) instead of moving average
sent_dual(text_polar, text_polar2, smooth=True, save=True, savefile='sent_across.png')

###############################
# sentiment analysis (split based on chapter numbering)
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

# visualize lower dimensions for topic vectors
# Dominant topic number in each doc
topic_num = np.argmax(doc_topic, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_nmf = tsne_model.fit_transform(doc_topic)

# Plot the Topic Clusters using Bokeh
TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save"
n_topics = 3
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

# create a df for tooltip assignment
doc_topic_p = pd.DataFrame({'topic': topic_num,
                            'topic_idx': range(1, doc_topic.shape[0] + 1),
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

# NMF on chapter numbering
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

#############################################################
# lda modeling (gensim)
# processing pipeline in function

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

#############################
# topic models by paragraph
with open('txt_data/to_the_lighthouse_1927.txt') as f:
    text = f.readlines()

text_para = [x.split('\n')[0] for x in text]

# filter out any paragraph that contain element of insufficient length
text_clean = [x for x in text_para if len(x)>50]
text_clean = [TextBlob(x).words for x in text_clean]

# remove punctuation and lower case all words
text_clean = [[re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower()) for x in doc] for doc in text_clean]
# remove words of length less than 4
text_clean = [[x for x in doc if len(x)>3] for doc in text_clean]

# re-forming words in doc into docs
docs = [(' ').join(x) for x in text_clean]

# custom stop words
vectorizer = TfidfVectorizer(stop_words=potential_stop_words4,
                             ngram_range=(1, 1),
                             max_df=0.9)
doc_word = vectorizer.fit_transform(docs)
doc_word.shape

# NMF model
nmf_model = NMF(5)
doc_topic = nmf_model.fit_transform(doc_word)
display_topics(nmf_model, vectorizer.get_feature_names(), 10)


