from nltk.tokenize import word_tokenize

# load data
filename = 'marcus_aurelius.txt'
file = open(filename, 'rt', encoding='utf-8')
text = file.read()
file.close()

# split into words
tokens = word_tokenize(text)

# save to file
with open('tokenized.txt', 'w', encoding='utf-8') as fp:
    for token in tokens:
        fp.write("%s\n" % token)
    print('Done')
