import string
import nltk
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# load data
filename = 'marcus_aurelius.txt'
file = open(filename, 'rt', encoding='utf-8')
text = file.read()
file.close()

# set up data processing
table = str.maketrans(","*len(string.punctuation), string.punctuation)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# split into sentence arrays
sentences = []
for sentence in nltk.sent_tokenize(text):
    words = nltk.word_tokenize(sentence)
    words = [w.lower() for w in words]
    words = [w.translate(table) for w in words]
    words = [word for word in words if word.isalnum()]
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    if words:
        sentences.append(words)

# train model
model = Word2Vec(sentences, min_count=20, sg=1, vector_size=1000)

# print top 10 words similar to 'greek'
sims = model.wv.most_similar('greek', topn=10)
print(sims)

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
