import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use TensorFlow's Keras API
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional, Input, Lambda, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import markovify
# import pronouncing
import os

# from transformers import BertTokenizer, TFBertModel
# from google.colab import drive
import re
import random

import nltk
import string
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import re
import sys
import io
import csv
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
# drive.mount('/content/drive')
path = './data/lyrics_rhyming_pairs_large.csv'

def ngram(token_list):
  ng = []
  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    ng.append(n_gram_sequence)
  return ng

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

path_paired_rhymes = "./grouped_rhymed.csv"
df = pd.read_csv(path_paired_rhymes)

df['text'] = df.first_line + " <linebreak> " + df.next_line + "<linebreakkk>"
df['alt_text'] = df.first_line + "\n" + df.next_line + "\n"
grouped_text = df.groupby('label')['text'].apply(''.join)
grouped_text_alt = df.groupby('label')['alt_text'].apply(''.join)
train = grouped_text.reset_index()
train_alt = grouped_text_alt.reset_index()
def concatenate_first_20_texts(group):
    return ' '.join(group)

# Group by 'group_col' and apply the function
result = df.groupby('label')['alt_text'].apply(concatenate_first_20_texts).reset_index()
str_lst = result['alt_text'].to_list()
st = "".join(result['alt_text'].to_list())

df['last_word']= [i[-1] for i in df['first_line'].str.split()]
df['target_word']= [i[-1] for i in df['next_line'].str.split()]
df['sec_to_last_word']= [i[-2] if len(i) > 1 else i[-1] for i in df['next_line'].str.split()]

df = df[df['sec_to_last_word'] != df['target_word']]

# ## 1. Combined Model and LSTM Simultaneously


X = (df['last_word'] + " " + df['sec_to_last_word']).to_list()
y = df['target_word'].to_list()

# Clean and lower case the text data
X_clean = [re.sub(r"\W+", ' ', text).lower().strip() for text in X]
y_clean = [re.sub(r"\W+", '', text).lower().strip() for text in y]

# Create and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_clean + y_clean)  # Fit on both X and y

# Convert text to sequences
X_seq = tokenizer.texts_to_sequences(X_clean)
y_seq = tokenizer.texts_to_sequences(y_clean)

# Check if X_seq and y_seq are aligned
if len(X_seq) != len(y_seq):
    raise ValueError("The lengths of X_seq and y_seq do not match.")

# Pad X sequences to a fixed length
max_length = 2  # Since you have two words per sequence in X
X_pad = pad_sequences(X_seq, maxlen=max_length, padding='post')

# Reshape X for LSTM
X_reshaped = np.reshape(X_pad, (len(X_pad), max_length, 1))

# Prepare y as one-hot encoded
total_words = len(tokenizer.word_index) + 1
y_one_hot = to_categorical([item for sublist in y_seq for item in sublist], num_classes=total_words)

# Split the data
split_idx = int(len(X_reshaped) * 0.8)  # 80% for training
X_train, X_test = X_reshaped[:split_idx], X_reshaped[split_idx:]
y_train, y_test = y_one_hot[:split_idx], y_one_hot[split_idx:]

model = Sequential()
model.add(Embedding(total_words, 150))
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


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback to save the model weights
checkpoint_filepath = 'lstm_model_checkpoint.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True
)

# Fit the model with the callback
model.fit(X_train, y_train, epochs=50, batch_size=64, callbacks=[model_checkpoint_callback])

# Save the tokenizer
# tokenizer.save('tokenizer.pkl')

# Save the labeled DataFrame
df.to_csv('labeled_data.csv', index=False)




combined_model = None
for i in range(0, len(str_lst), 100):
  tm = markovify.Text(str_lst[i:i+100])
  if combined_model:
      combined_model = markovify.combine(models=[combined_model, tm])
  else:
      combined_model = tm

