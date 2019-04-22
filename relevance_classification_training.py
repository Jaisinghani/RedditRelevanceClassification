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
import pickle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

nltk.download('stopwords')


def applyMultipleLayers(input_, layers):
    if not len(layers) > 1:
        raise ValueError('This list should contain 2 or more layers to be used')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_



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
    return 2*((precision*recall)/(precision+recall))

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    text = text.lower().split()
    if remove_stopwords:
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
    

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return(text)

# Initialize global variables
DATASETS_DIR = ''
QUORA_QUESTION_FILE = 'quora_updated_dataset.txt'
GLOVE_FILE = 'glove.840B.300d.txt'
C1_TRAINING_DATA_FILE = 'c1_train_bilstm.npy'
C2_TRAINING_DATA_FILE = 'c2_train_bilstm.npy'
LABEL_TRAINING_DATA_FILE = 'label_train_bilstm.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix_bilstm.npy'
NB_WORDS_DATA_FILE = 'nb_words_bilstm.json'
MAX_SEQUENCE_LENGTH = 30 
MODEL_WEIGHTS_FILE = 'model_pairs_weights.h5.bilstm.withattention'
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
NB_EPOCHS = 25
DROPOUT = 0.1 #Hyperparameter
BATCH_SIZE = 500 #Hyperparameter
WORD_EMBEDDING_DIM = 300
SENT_EMBEDDING_DIM = 200
RNG_SEED = 13371447
OPTIMIZER = 'adam'
MAX_NB_WORDS = 2000000
REDDIT_DATA_FILE = 'reddit_test_data_excel.txt'

def readTestFile():
    sentence1 = []
    sentence2 = []

    is_relevant = []
    c = 0
    with open(DATASETS_DIR + REDDIT_DATA_FILE) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            sentence1.append(text_to_wordlist(row['sentence1'], remove_stopwords=True, stem_words=True))
            sentence2.append(text_to_wordlist(row['sentence2'], remove_stopwords=True, stem_words=True))
            is_relevant.append(row['is_relevant'])

    sentences = sentence1 + sentence2
    return sentences

# If the dataset, embedding matrix and word count exist in the local directory
if exists(C1_TRAINING_DATA_FILE) and exists(C2_TRAINING_DATA_FILE) and exists(LABEL_TRAINING_DATA_FILE) and exists(NB_WORDS_DATA_FILE) and exists(WORD_EMBEDDING_MATRIX_FILE):
    # Then load them
    q1_data = np.load(open(C1_TRAINING_DATA_FILE, 'rb'))
    q2_data = np.load(open(C2_TRAINING_DATA_FILE, 'rb'))
    labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))
    word_embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
    with open(NB_WORDS_DATA_FILE, 'r') as f:
        nb_words = json.load(f)['nb_words']
else:

    print("Processing", QUORA_QUESTION_FILE)

    sentence1 = []
    sentence2 = []

    is_relevant = []
    c = 0
    with open(DATASETS_DIR + QUORA_QUESTION_FILE) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            sentence1.append(text_to_wordlist(row['sentence1'], remove_stopwords=True, stem_words=True))
            sentence2.append(text_to_wordlist(row['sentence2'], remove_stopwords=True, stem_words=True))
            is_relevant.append(row['is_relevant'])

    print('text pairs: %d' % len(sentence1))


    redditData = readTestFile()
    sentences = sentence1 + sentence2 + redditData
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(sentences)
    sentence1_word_sequences = tokenizer.texts_to_sequences(sentence1)
    sentence2_word_sequences = tokenizer.texts_to_sequences(sentence2)

    word_index = tokenizer.word_index
    #print("word_index :", word_index)



    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)




    print("Words in index: %d" % len(word_index))

    # Download and process GloVe embeddings
    # if not exists(KERAS_DATASETS_DIR + GLOVE_ZIP_FILE):
    #     zipfile = ZipFile(get_file(GLOVE_ZIP_FILE, GLOVE_ZIP_FILE_URL))
    #     zipfile.extract(GLOVE_FILE, path=KERAS_DATASETS_DIR)

    print("Processing", GLOVE_FILE)

    embeddings_index = {}
    with open(DATASETS_DIR + GLOVE_FILE) as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    print('Word embeddings: %d' % len(embeddings_index))

    # Prepare word embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, WORD_EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
        else:
            word_embedding_matrix[i] = np.random.rand(1,300)

        
    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

    # Prepare training data tensors
    q1_data = pad_sequences(sentence1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    #print("q1_data :",q1_data)
    q2_data = pad_sequences(sentence2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    #print("q2_data :", q2_data)
    labels = np.array(is_relevant, dtype=int)
    print('Shape of sentence1 data tensor:', q1_data.shape)
    print('Shape of sentence2 data tensor:', q2_data.shape)
    print('Shape of label tensor:', labels.shape)

    # Persist training and configuration data to files
    np.save(open(C1_TRAINING_DATA_FILE, 'wb'), q1_data)
    np.save(open(C2_TRAINING_DATA_FILE, 'wb'), q2_data)
    np.save(open(LABEL_TRAINING_DATA_FILE, 'wb'), labels)
    np.save(open(WORD_EMBEDDING_MATRIX_FILE, 'wb'), word_embedding_matrix)
    with open(NB_WORDS_DATA_FILE, 'w') as f:
        json.dump({'nb_words': nb_words}, f)

sys.stdout.flush()
# Partition the dataset into train and test sets
X = np.stack((q1_data, q2_data), axis=1)
#print("X:", X)
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]

print("Q1_train.shape: ", Q1_train.shape)
print("Q2_train.shape :", Q2_train.shape)
print("Q1_test.shape: ", Q1_test.shape)
print("Q2_test.shape :", Q2_test.shape)

# Define the model
sentence1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
sentence2 = Input(shape=(MAX_SEQUENCE_LENGTH,))
q1 = Embedding(nb_words + 1, 
                 WORD_EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(sentence1)
print("q1 shape :", q1.shape)
q1 = Bidirectional(LSTM(SENT_EMBEDDING_DIM, return_sequences=True))(q1)

print("q1 shape :", q1.shape)
q2 = Embedding(nb_words + 1, 
                 WORD_EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(sentence2)
print("q2 shape :", q2.shape)
q2 = Bidirectional(LSTM(SENT_EMBEDDING_DIM, return_sequences=True))(q2)


q1_rep = applyMultipleLayers(q1, [GlobalAvgPool1D(), GlobalMaxPool1D()])
q2_rep = applyMultipleLayers(q2, [GlobalAvgPool1D(), GlobalMaxPool1D()])


merged = concatenate([q1_rep, q2_rep])
merged = BatchNormalization()(merged)
merged = Dense(1000, activation='elu')(merged)

merged = BatchNormalization()(merged)
merged = Dense(500, activation='elu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='elu')(merged)
merged = BatchNormalization()(merged)
merged = Dense(100, activation='elu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)

is_relevant = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[sentence1,sentence2], outputs=is_relevant)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
print("Starting training at", datetime.datetime.now())
sys.stdout.flush()
t0 = time.time()
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
history = model.fit([Q1_train, Q2_train],
                    y_train,
                    epochs=NB_EPOCHS,
                    validation_split=VALIDATION_SPLIT,
                    verbose=2,
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks)
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

# Print best validation accuracy and epoch
max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
print('Maximum validation accuracy = {0:.4f} (epoch {1:d})'.format(max_val_acc, idx+1))

# Evaluate the model with best validation accuracy on the test partition
model.load_weights(MODEL_WEIGHTS_FILE)
loss, accuracy, f1 = model.evaluate([Q1_test, Q2_test], y_test, verbose=0)
print(model.evaluate([Q1_test, Q2_test], y_test, verbose=0))
print('Test loss = {0:.4f}, test accuracy = {1:.4f}, F1 score = {1:.4f}'.format(loss, accuracy, f1))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
sys.stdout.flush()

