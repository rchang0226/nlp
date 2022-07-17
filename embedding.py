from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy as np

# read tokenized words from file
words = []
with open('C:/Users/raymo/OneDrive/Documents/nlp/tokenized.txt', 'r', encoding='utf-8') as fp:
    for line in fp:
        # remove linebreak from each word (last character)
        x = line[:-1]
        words.append(x)

# split into sentence arrays
sentences = []
sentence = []
for word in words:
    if word == '.':
        sentences.append(sentence)
        sentence = []
    elif word.isalnum():
        sentence.append(word)

print(sentences)

# train model
model = Word2Vec(sentences, min_count=1)
# fit a 2d pca model to the vectors
X = np.asarray(model.wv.vectors)
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.index_to_key)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
