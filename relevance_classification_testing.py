from __future__ import print_function

import csv
import re

import nltk
from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from numpy import *

nltk.download('stopwords')
DATASETS_DIR = ''
REDDIT_DATA_FILE = 'reddit_test_data_excel.txt'
MAX_NB_WORDS = 2000000
MAX_SEQUENCE_LENGTH = 30
MODEL_WEIGHTS_FILE = 'model_pairs_weights.h5.bilstm.withattention'

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        print("true :", true_positives)
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    # def confusionmatrix(y_pred, y_true):
    #     row = {}
    #     for i in range(len(y_true)):
    #         x = str(y_true[i]) + str(y_pred[i])
    #         key = "group_{0}".format(x)
    #         if key in row:
    #             row["group_{0}".format(x)] = row["group_{0}".format(x)] + 1
    #         else:
    #             row["group_{0}".format(x)] = 1
    #
    #     labelrows = []
    #     for x in range(0, 2):
    #         for y in range(0, 2):
    #             j = str(x) + str(y)
    #             p = "group_{0}".format(j)
    #             if p in row:
    #                 labelrows.append(row["group_{0}".format(j)])
    #             else:
    #                 labelrows.append(0)
    #
    #     cm = reshape(labelrows, (2, 2))
    #     true_positive = cm[1][1]
    #     false_positive = cm[0][1]
    #     true_negative = cm[0][0]
    #     false_negative = cm[1][0]
    #     return true_positive, false_positive, true_negative, false_negative
    #
    # true_positive, false_positive, true_negative, false_negative = confusionmatrix(y_pred, y_true)
    #
    # precision = round(float(true_positive / (true_positive + false_positive)), 5)
    #
    # recall = round(float(true_positive / (true_positive + false_negative)), 5)
    #
    # tpr = round(float(true_positive / (true_positive + false_negative)), 5)
    #
    # fpr = round(float(false_positive / (true_negative + false_positive)), 5)
    #
    # tnr = round(float(true_negative / (true_negative + false_positive)), 5)
    #
    # fnr = round(float(false_negative / (true_positive + false_negative)), 5)
    #
    # print("precision :", precision)
    # print("recall :", recall)
    # print("True positive rate  :", tpr)
    # print("True negative rate  :", tnr)
    # print("False positive rate  :", fpr)
    # print("False negative rate  :", fnr)
    #
    # return (precision, recall, tpr, fpr, tnr, fnr)

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def cleanup(text, removeStopwords=False, stemWords=False):

    text = text.lower().split()

    if removeStopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)


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
    if stemWords:
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
        sentence1.append(cleanup(row['sentence1'], removeStopwords=True, stemWords=True))
        sentence2.append(cleanup(row['sentence2'], removeStopwords=True, stemWords=True))
        is_relevant.append(row['is_relevant'])

print('data pairs: %d' % len(sentence1))




sentences = sentence1 + sentence2

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
tokenizer.fit_on_texts(sentences)
sentence1_word_sequences = tokenizer.texts_to_sequences(sentence1)

sentence2_word_sequences = tokenizer.texts_to_sequences(sentence2)

word_index = tokenizer.word_index


print("Words in index: %d" % len(word_index))

c1_data = pad_sequences(sentence1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)

c2_data = pad_sequences(sentence2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = np.array(is_relevant, dtype=int)
print('Shape of sentence1 data tensor:', c1_data.shape)
print('Shape of sentence2 data tensor:', c2_data.shape)
print('Shape of label tensor:', labels.shape)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])


loss, accuracy, f1 = loaded_model.evaluate([c1_data, c2_data], labels, verbose=0)
print(loaded_model.evaluate([c1_data, c2_data], labels, verbose=0))
print('Test loss = {0:.4f}, test accuracy = {1:.4f}, F1 score = {1:.4f}'.format(loss, accuracy, f1))



def CosineSimilarity(vector, otherPhraseVec):
		cosine_similarity = np.dot(vector, otherPhraseVec) / (np.linalg.norm(vector) * np.linalg.norm(otherPhraseVec))
		try:
			if math.isnan(cosine_similarity):
				cosine_similarity=0
		except:
			cosine_similarity=0
		return cosine_similarity

similarityScores = []
print(type(c1_data))
print(c1_data.shape)
for i in range(len(c1_data)):
    sim = CosineSimilarity(c1_data[i], c2_data[i])
    similarityScores.append(sim)
