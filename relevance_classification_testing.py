from __future__ import print_function
import numpy as np
import sys, csv, datetime, time, json
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import * #Dot, Input, Bidirectional, GRU,LSTM, dot, Flatten, Dense, Reshape, add, Dropout, BatchNormalization, concatenate, Lambda, Permute, Concatenate, Multiply
from keras.activations import softmax
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from string import punctuation
import nltk
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.models import model_from_json
import pickle

nltk.download('stopwords')
DATASETS_DIR = ''
REDDIT_DATA_FILE = 'reddit_test_data.txt'
MAX_NB_WORDS = 2000000
MAX_SEQUENCE_LENGTH = 30
MODEL_WEIGHTS_FILE = 'model_pairs_weights.h5.bilstm.withattention'


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # text = unicode(text, "utf-8")
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    # Return a list of words
    return (text)

sentence1 = []
sentence2 = []

is_relevant = []
c = 0

with open(DATASETS_DIR + REDDIT_DATA_FILE) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
        print(row.keys())
        sentence1.append(text_to_wordlist(row['sentence1'], remove_stopwords=True, stem_words=True))
        sentence2.append(text_to_wordlist(row['sentence2'], remove_stopwords=True, stem_words=True))
        is_relevant.append(row['is_relevant'])

print('data pairs: %d' % len(sentence1))



# Build tokenized word index
sentences = sentence1 + sentence2

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
tokenizer.fit_on_texts(sentences)
sentence1_word_sequences = tokenizer.texts_to_sequences(sentence1)

sentence2_word_sequences = tokenizer.texts_to_sequences(sentence2)

word_index = tokenizer.word_index
#print("word_index :", word_index)

print("Words in index: %d" % len(word_index))

q1_data = pad_sequences(sentence1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
# print("q1_data :",q1_data)
q2_data = pad_sequences(sentence2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
# print("q2_data :", q2_data)
labels = np.array(is_relevant, dtype=int)
print('Shape of sentence1 data tensor:', q1_data.shape)
print('Shape of sentence2 data tensor:', q2_data.shape)
print('Shape of label tensor:', labels.shape)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
# Evaluate the model with best validation accuracy on the test partition

loss, accuracy, f1 = loaded_model.evaluate([q1_data, q2_data], labels, verbose=0)
print(loaded_model.evaluate([q1_data, q2_data], labels, verbose=0))
print('Test loss = {0:.4f}, test accuracy = {1:.4f}, F1 score = {1:.4f}'.format(loss, accuracy, f1))

