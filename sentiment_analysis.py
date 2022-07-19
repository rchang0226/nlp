import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import string
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
import numpy as np


def load_data(path):
    text = []
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    for filename in files:
        file = open(os.path.join(path, filename), 'rt', encoding='utf-8')
        doc = file.read()
        file.close()
        text.append(doc)
    return text


def tokenize_text(text):
    table = str.maketrans("," * len(string.punctuation), string.punctuation)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    words = nltk.word_tokenize(text)
    words = [w.lower() for w in words]
    words = [w.translate(table) for w in words]
    words = [word for word in words if word.isalnum()]
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return words


# run this to generate the 'vocab.txt' file
def create_vocab(data):
    # create and add to vocab
    vocab = Counter()
    for tokens in data:
        vocab.update(tokens)
    print(len(vocab))
    print(vocab.most_common(50))

    # filter out tokens with low occurrence
    tokens = [k for k, c in vocab.items() if c >= 2]
    print(len(tokens))

    # save tokens to file
    with open('vocab.txt', 'w', encoding='utf-8') as fp:
        for token in tokens:
            fp.write("%s\n" % token)
        print('Done')


def load_vocab(filename):
    words = []
    with open(filename, 'r') as fp:
        for line in fp:
            x = line[:-1]
            words.append(x)
    return words


# set up data and split into test and train
pos_text = load_data('txt_sentoken/pos')
pos_train = pos_text[:-100]
pos_test = pos_text[-100:]

neg_text = load_data('txt_sentoken/neg')
neg_train = neg_text[:-100]
neg_test = neg_text[-100:]

all_train = pos_train + neg_train
all_test = pos_test + neg_test

# create the labels
ytrain = np.array([1] * 900 + [0] * 900)
ytest = np.array([1] * 100 + [0] * 100)

# tokenize the texts
train_cleaned = [tokenize_text(text) for text in all_train]
test_cleaned = [tokenize_text(text) for text in all_test]

# load vocab
vocab = set(load_vocab('vocab.txt'))

# filter out tokens not in vocab
train = [[w for w in tokens if w in vocab] for tokens in train_cleaned]
test = [[w for w in tokens if w in vocab] for tokens in test_cleaned]

# set up tokenizer
t = Tokenizer()
t.fit_on_texts(train)
vocab_size = len(t.word_index) + 1

# encode
encoded = t.texts_to_sequences(train)
encoded_test = t.texts_to_sequences(test)

# pad sequences
max_length = max([len(s) for s in train])
Xtrain = pad_sequences(encoded, maxlen=max_length, padding='post')
Xtest = pad_sequences(encoded_test, maxlen=max_length, padding='post')

# define the model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize the model
print(model.summary())

# fit the model
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate the model
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
