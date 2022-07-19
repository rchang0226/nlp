import string
import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


# define the training data
sentences = [
    "Tons of fun!!!",
    "Potential is massive, but without mods it gets boring, and constantly updating mods every time new beta comes "
    "out is tedious",
    "Game is really good, it is a huge improvement from warband but it lacks some features warband had like assigning "
    "your own formation or feasts but its not a big deal",
    "One of the best war games I've played plus I used voice attack for voice commands",
    "The main game loop feels pretty polished and satisfying",
    "The game is fun, no doubt about it",
    "Sieges have various issues",
    "Executions are a questionable feature",
    "Some mechanics don't work, or work poorly",
    "Troop trees are rough and basic and copy-paste vanilla values, uninteresting to explore"
]

# class labels (1 for positive reception, 0 for negative)
labels = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0])

# set up data processing
table = str.maketrans(","*len(string.punctuation), string.punctuation)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# split into tokenized sentence list
cleaned = []
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    words = [w.lower() for w in words]
    words = [w.translate(table) for w in words]
    words = [word for word in words if word.isalnum()]
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    if words:
        cleaned.append(words)

# set up tokenizer
t = Tokenizer()
t.fit_on_texts(cleaned)
vocab_size = len(t.word_index) + 1

# encode text
encoded = t.texts_to_sequences(cleaned)
print(encoded)

# pad all to same length
max_length = max(len(s) for s in cleaned)
print(max_length)
padded = pad_sequences(encoded, maxlen=max_length, padding='post')
print(padded)

# define the model
model = Sequential()
model.add(Embedding(vocab_size, 16, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())

# fit the model
model.fit(padded, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))

