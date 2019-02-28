#########################
# LDA model (need count vectorizer, not tf-idf) first try
doc_word = doc_word.transpose()
# pd.DataFrame(doc_word.toarray(), vectorizer.get_feature_names()).head()

# Convert sparse matrix of counts to a gensim corpus
corpus = matutils.Sparse2Corpus(doc_word)
id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())

lda = models.LdaModel(corpus=corpus, num_topics=3, id2word=id2word, passes=5)
lda.print_topics()

#########################
# try clustering on noun phrases
text_blob = TextBlob(' '.join(text_clean)).noun_phrases

# exclude noun_phrases that include stop words
text_blob_new = []
for item in text_blob:
    if all(subitem not in potential_stop_words for subitem in item.split()):
        text_blob_new.append(item)

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

###########################
# text frequencies for noun phrases that don't include stop words
text_clean_str = ' '.join(text)
text_clean_blob = TextBlob(text_clean_str).noun_phrases
text_clean_blob_temp = []
for item in text_clean_blob:
    if all(subitem not in stop_words_v5 for subitem in item.split()):
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
wc = WordCloud(width=1600, height=800, max_words=30).generate_from_frequencies(freq_dict2)
plt.figure(figsize=(20, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('lighthouse_wc.png', facecolor='k', bbox_inches="tight")
plt.show()

#########################
# plot with twin axes of different scales
t = np.arange(0.01, 10.0, 0.01)
data1 = np.exp(t)
data2 = np.sin(2 * np.pi * t)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('exp', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# topic results (5 documents, 4 topics, 0.8 max_df)
# Topic  0
# brush, grass, green, summer, silence, sent, canvas, boots, shore, wall
# Topic  1
# love, watch, course, dish, true, listening, french, sort, happened, plate
# Topic  2
# blundered, severity, urn, blame, odious, circus, tobacco, hope, leg, stopping
# Topic  3
# lights, lost, flounder, grow, love, reason, necklace, bit, oneself, brooch

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
