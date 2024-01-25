import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Use TensorFlow's Keras API
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional, Input, Lambda, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


import re
import random
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
# drive.mount('/content/drive')
# path = '/content/drive/My Drive/lyrics_rhyming_pairs.csv'


import pandas as pd
import nltk
import string
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import re
import sys
import io
import csv


class Annotator:
    def __init__(self):
        self.__phoneme_table = self.__create_phoneme_table()

    def __create_phoneme_table(self):
        # the phoneme table is a python dictionary of indexed by the
        # word and the value is the a list of list of phonemes.
        # the elements in the outer list correspond to alternate
        # pronounciations of the word, and the inner list is the
        # sequence of phonemes corresponding to that word in the
        # CMU Pronouncing Dictionary
        table = dict()

        with open("./data/processing/cmu_pronuncing_dict/cmudict-0.7b", 'r', encoding="iso8859_2") as cmu_dict_file:
            for i in range(126):
                next(cmu_dict_file)
            while True:
                try:
                    line = next(cmu_dict_file).split()
                except StopIteration:
                    break
                line[0] = re.sub(r'\([0-9]\)', '', line[0])
                if line[0] in table.keys():
                    table[line[0]].append(line[1:])
                else: table[line[0]] = [line[1:]]
        return table

    # expects lyrics as a list of sentences
    def find_rhyme_pairs(self, lyrics):
        # pairs is a list of tuples, the tuples represent pairs of indices
        # corresponding to lines that rhyme together
        pairs = list()
        for i in range(len(lyrics)-1):
            line = lyrics[i].split()
            for j in range(1,3):
                try:
                    next = lyrics[i+j].split()
                    if self.is_rhyme(line[-1], next[-1]):
                        pairs.append((i,i+j))
                except Exception: continue
        return pairs

    def annotate_rhyme_pairs(self, poems_data_row):
        return self.find_rhyme_pairs(poems_data_row["Content"].split('\n'))

    def annotate_POS_tags(self, poems_data_row):
        return [pos_tag(word_tokenize(line)) for line in poems_data_row["Content"].split('\n')]

    def get_phonemes_list(self, word):
        return self.__phoneme_table[word.upper()]

    # if ignore_stress=False, finds the last syllable in the word with
    # primary or secondary stress, and returns true if this syllable
    # and the ones following it match those in the otherword
    # if ignore_stress=True, only finds the last syllable regardless of
    # stress and matches it to the other word's.
    def is_rhyme(self, word, otherword, ignore_stress=False):
        include_stress = ['1','2']
        if ignore_stress: include_stress.append('0')

        word = word.upper().strip(string.punctuation)
        otherword = otherword.upper().strip(string.punctuation)
        # A word cannot rhyme with itself
        if word == otherword: return False

        try:
            wordphonemes_list = self.get_phonemes_list(word)
            otherwordphonemes_list = self.get_phonemes_list(otherword)
        # if the word is not in CMU dictionary return False
        except KeyError: return False

        # wordphonemes_list and otherwordphonemes_list are lists of lists
        # of phonemes retrieved from the table, for different pronounciations
        # of each word.
        for wordphonemes in wordphonemes_list:
            for otherwordphonemes in otherwordphonemes_list:
                ind = 0
                for i in reversed(range(len(wordphonemes))):
                    # checks if the phoneme has character 1 or 2 at the end, i.e.
                    # if it's a vowel with primary or secondary stress
                    if wordphonemes[i][-1] in include_stress:
                        ind = i
                        break
                rhymephonemes = wordphonemes[ind:]
                if otherwordphonemes[-len(rhymephonemes):] == rhymephonemes: return True
        return False
    def is_rhyme_for_col(self, word, otherword, ignore_stress=False):
        include_stress = ['1','2']
        if ignore_stress: include_stress.append('0')

        word = word.upper().strip(string.punctuation)
        otherword = otherword.upper().strip(string.punctuation)
        # A word cannot rhyme with itself
        if word == otherword: return True

        try:
            wordphonemes_list = self.get_phonemes_list(word)
            otherwordphonemes_list = self.get_phonemes_list(otherword)
        # if the word is not in CMU dictionary return False
        except KeyError: return False

        # wordphonemes_list and otherwordphonemes_list are lists of lists
        # of phonemes retrieved from the table, for different pronounciations
        # of each word.
        for wordphonemes in wordphonemes_list:
            for otherwordphonemes in otherwordphonemes_list:
                ind = 0
                for i in reversed(range(len(wordphonemes))):
                    # checks if the phoneme has character 1 or 2 at the end, i.e.
                    # if it's a vowel with primary or secondary stress
                    if wordphonemes[i][-1] in include_stress:
                        ind = i
                        break
                rhymephonemes = wordphonemes[ind:]
                if otherwordphonemes[-len(rhymephonemes):] == rhymephonemes: return True
        return False

      
  


def decode_content(poem_data_row):
    poem_data_row["Content"] = unidecode.unidecode(poem_data_row["first_line"])
    return poem_data_row

#with open("C:\\Users\\chenruiyang\\Desktop\\NLP\\final prject\\lyrics_ryhming_pairs.csv", 'r', encoding="iso8859_2") as f:


#def grouping(file):
def write_to_csv(data, file_name):
    with open(file_name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        for idx, group in enumerate(data):
            if idx > 0:  # Write a blank line if not the first group
                writer.writerow([])

            for line in group:
                writer.writerow([line])




def ngram(token_list):
  ng = []
  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    ng.append(n_gram_sequence)
  return ng

df = pd.read_csv("./grouped_rhymed.csv")
df['text'] = df.first_line + " linebreak " + df.next_line + " linebreak "
grouped_text = df.groupby('label')['text'].apply(' linebreak '.join)

# Convert to DataFrame
train = grouped_text.reset_index()['text']

# Initialize Tokenizer
tokenizer = Tokenizer()

# Preprocess text
corpus = [text for text in train if isinstance(text, str) and text.strip()]

# Fit tokenizer on the corpus
tokenizer.fit_on_texts(corpus)
max_sequence_len = 50
# Generate input sequences using n-grams
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, min(len(token_list), max_sequence_len)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Split data into predictors and label
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

# Total number of words
total_words = len(tokenizer.word_index) + 1
fin_data = pd.DataFrame(np.hstack((predictors, label.reshape(-1,1))),columns=np.hstack((np.arange(1,predictors.shape[1]+1),np.array(['label']))))
total_words = len(tokenizer.word_index) + 1

fdf = pd.DataFrame(np.hstack((predictors, label.reshape(-1,1))),columns=np.hstack((np.arange(1,predictors.shape[1]+1),np.array(['label']))))

n_col_m_1 = fdf.shape[1]-1
dataX = [fdf.iloc[i,0: n_col_m_1].tolist() for i in range(0,fdf.shape[0])]
proportion = (int)(len(dataX) * 0.8)
dataX_train = dataX[:proportion]
dataX_test = dataX[proportion:]


dataY = [fdf.iloc[i,n_col_m_1] for i in range(0,fdf.shape[0])]
dataY_train = dataY[:proportion]
dataY_test = dataY[proportion:]
print(len(dataX_train))
print(len(dataX_test))
print(len(dataY_train))
print(len(dataY_test))


# reshape X to be [samples, time steps, features]
X_train = np.reshape(dataX_train, (len(dataX_train), n_col_m_1, 1))

# one hot encode the output variable
y_train = to_categorical(dataY_train)

# reshape X to be [samples, time steps, features]
X_test = np.reshape(dataX_test, (len(dataX_test), n_col_m_1, 1))

# one hot encode the output variable
y_test = to_categorical(dataY_test)

model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_len-1))
# Add an LSTM Layer
model.add(Bidirectional(LSTM(50, return_sequences=True)))
# A dropout layer for regularisation
model.add(Dropout(0.2))
# Add another LSTM Layer
model.add(LSTM(30))
model.add(Dense(total_words/2, activation='relu'))
# In the last layer, the shape should be equal to the total number of words present in our corpus
model.add(Dense(y_train.shape[1], activation='softmax'))
#model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')  #(# Pick a loss function and an optimizer)
print(model.summary())


early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=25, batch_size=64, callbacks=[early_stopping])

### Save model 
model.save('./my_small_lstm_model.h5')
model.save_weights('./my_small_lstm_weights.h5')
