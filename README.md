# Description
A collection of NLP exercises. 

## Tokenization
tokenization.py loads a txt file and tokenizes the document, creating a file with each word/token on a new line. Marcus Aurelius' Meditations was used as the text to be tokenized. The result is in tokenized.txt.

## Embedding
embedding.py trains a word embedding distributed representation. The training text was pre-processed to use stem words, filter out stop words, and more. Once again, Meditations was used as the training text. The 10 words most similar in context to "greek" were generated using this embedding, along with their relative distances: 

[('ancient', 0.9997207522392273), ('studi', 0.9997172951698303), ('father', 0.9997162818908691), ('learn', 0.9997159242630005), ('children', 0.9997142553329468), ('appli', 0.9997129440307617), ('found', 0.9997126460075378), ('number', 0.9997105002403259), ('came', 0.9997091889381409), ('master', 0.9997069835662842)]. 

The script also generates a plot to represent the embedding, projected onto 2 dimensions using PCA. 

## Learned Embedding
learned_embedding.py uses a Keras Embedding layer to classify text into two categories. The text used comes from Steam reviews of the game Mount and Blade II: Bannerlord. The reviews were classified as "recommend" and "not recommend." The text was tokenized, encoded, and padded to the same length prior to being fed into the network. 

## Sentiment Analysis
sentiment_analysis.py generates a model that classifies reviews as "positive" or "negative." The model was trained on the Movie Review Polarity Dataset created by Bo Pang and Lillian Lee. 

After splitting the data into train and test sets and tokenizing the documents, a vocabulary was defined by filtering out the tokens with low occurence. This vocab set was saved to vocab.txt. 

After filtering out tokens not in the vocab set, the data was encoded and padded before being fed into the model, the main features of which are the embedding layer and the CNN. The model is saved to 'model.h5'

The model achieves an accuracy of around 85% on the test dataset.
