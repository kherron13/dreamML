from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import find
import hdbscan
import numpy as np

ldreams = []
identifiers = []

with open('lemmatized.txt', encoding='ascii') as lemmatized:
    for i, line in enumerate(lemmatized):
        if i % 3 == 0:
            identifiers.append(line[:-1])
        elif (i - 1) % 3 == 0:
            ldreams.append(line)

cv = CountVectorizer(stop_words = 'english', min_df = 10, ngram_range = (1, 4))
tfidf = TfidfTransformer()
X = tfidf.fit_transform(cv.fit_transform(ldreams))
nonzeros = find(X)
#access words in vocabulary by feature index
vocab_by_index = dict((index, word) for word, index in cv.vocabulary_.items())

MIN_SAMPLES = 5
hd = hdbscan.HDBSCAN(metric = 'cosine', min_cluster_size = 8,\
                     min_samples = MIN_SAMPLES, approx_min_span_tree = False,\
                     alpha = 0.8, cluster_selection_method = 'leaf')
labels = hd.fit_predict(X)

#exclude noise label to get number of clusters
label_set = set(labels)
cl_count = len(label_set) - 1 if -1 in label_set else len(label_set)

#count occurrences of each feature for each cluster
feature_counts = [np.zeros(len(cv.vocabulary_)) for i in range(cl_count)]
for i, sample in enumerate(nonzeros[0]):
    if labels[sample] == -1:
        continue
    feature_counts[labels[sample]][nonzeros[1][i]] += 1

sample_indeces = [[] for i in range(cl_count)]
for i in range(len(labels)):
    if labels[i] == -1:
        continue
    sample_indeces[labels[i]].append(i)

for label in range(cl_count):
    
    defining_features = []          
    for i, f_count in enumerate(feature_counts[label]):
        if f_count >= MIN_SAMPLES - 1:
            defining_features.append(i)
    if len(defining_features) > 20:
        defining_features = np.argsort(feature_counts[label])[-20:][::-1]
    
    common_words = list(map(lambda x: vocab_by_index[x], defining_features))
    
    samples = list(map(lambda x: identifiers[x], sample_indeces[label]))
    
    print(label)
    print("Common features:\n", common_words)
    print("Dreams:\n", samples)

#print(len(list(filter(lambda x: x == -1, labels))) / len(labels)) # % of samples labeled as noise
