import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

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

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def calculate_perplexity(sentence, model_name='gpt2'):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Encode text inputs
    tokens = tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt")

    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Avoid calculating gradients
    with torch.no_grad():
        # Calculate loss. This outputs the average log-likelihood of the sequence
        outputs = model(tokens, labels=tokens)
        loss = outputs[0]

    # Calculate perplexity
    perplexity = torch.exp(loss).item()

    return perplexity


random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


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

        with open("/content/drive/My Drive/cmudict-0.7b", 'r', encoding="iso8859_2") as cmu_dict_file:
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


def concatenate_first_20_texts(group):
    return ' '.join(group)



def make_lstm():
  model = Sequential()
  model.add(Embedding(total_words, 150))
  # Add an LSTM Layer
  model.add(Bidirectional(LSTM(50, return_sequences=True)))
  # A dropout layer for regularisation
  model.add(Dropout(0.2))
  # Add another LSTM Layer
  model.add(LSTM(30))
  # model.add(Dense(total_words/2, activation='relu'))
  # In the last layer, the shape should be equal to the total number of words present in our corpus
  model.add(Dense(y_train.shape[1], activation='softmax'))
  return model

#model.add(Dense(total_words, activation='softmax'))


def make_bar(bar_len=30):
  bar = None
  while (bar == None):
    bar = text_model.make_short_sentence(bar_len, max_overlap_ratio = .49, tries=400)
  return bar

def make_rhymes(text_):

  prev_word = text_.split()[-1]
  ann = Annotator()
  flag = False
  count = 0
  while flag != True:
    count += 1
    bar = make_bar()


    new_bar = bar.split()
    new_bar = [re.sub(r"\W+", '', i).lower() for i in new_bar]
    new_bar = " ".join(new_bar)
    last_word = new_bar.split()[-2]
    bar = bar.split()


    if (ann.is_rhyme(new_bar[-1], prev_word)):
      print("Input Text:", text_)
      print("Response Text:", bar)
      print()
      print("Naturally Rhymed and count is", count)
      flag = True
      break

    token_list = tokenizer.texts_to_sequences([prev_word.lower(), last_word.lower()])

    # Flatten token_list and pad to max_length
    token_list = [item for sublist in token_list for item in sublist]
    token_list_padded = pad_sequences([token_list], maxlen=max_length, padding='post')

    # Reshape to (1, max_length, 1)
    token_list_reshaped = np.reshape(token_list_padded, (1, max_length, 1))

    predicted = model.predict(token_list_reshaped, verbose=0)
    predicted_index =  np.argmax(predicted)

    #predicted_index=1
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word

            break
    if (ann.is_rhyme(output_word, prev_word)):
      bar[-1] = output_word
      bar = " ".join(bar)
      print("Input Text:", text_)
      print("Response Text:", bar)
      print()
      print("Rhymed and count is", count)
      flag = True

    if count == 50:

      print("No Rhyme")
      print(text_)
      print(bar)
      flag = True
  return text_, bar


rhymed_pairs_path = "./lyrics_rhyming_pairs_large.csv"
rhymed_pairs = pd.read_csv(rhymed_pairs_path)


print(rhymed_pairs.head())
rhymed_results = []

for index, row in rhymed_pairs.iterrows():
    result_first_line = make_rhymes(row['first_line'], annotator, model, tokenizer)
    result_next_line = make_rhymes(row['next_line'], annotator, model, tokenizer)
    
    rhymed_results.extend([result_first_line, result_next_line])

# Create a DataFrame from the results
rhymed_results_df = pd.DataFrame(rhymed_results, columns=['Input Text', 'Response Text'])

# Save the results to a CSV file
rhymed_results_df.to_csv("./rhymed_results.csv", index=False)
import csv

# Load sentences from the CSV file
csv_filename = "./data/rhymed_results.csv"
sentences_to_evaluate = []

with open(csv_filename, mode='r', newline='') as file:
    reader = csv.reader(file)
    
    next(reader)

    for row in reader:
        sentences_to_evaluate.append(row[1])
output_csv_filename = "results.csv"
with open(output_csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(["Sentence", "Perplexity", "Flesch-Kincaid Grade Level", "Gunning Fog Index"])
    

    for sentence in sentences_to_evaluate:
        perplexity = calculate_perplexity(sentence)
        fk_grade = textstat.flesch_kincaid_grade(sentence)
        gunning_fog = textstat.gunning_fog(sentence)
        
        print(f"Sentence: \"{sentence}\"")
        print(f"Perplexity: {perplexity}")
        print(f"Flesch-Kincaid Grade Level: {fk_grade}")
        print(f"Gunning Fog Index: {gunning_fog}")
        print("\n")
    
        writer.writerow([sentence, perplexity, fk_grade, gunning_fog])

print(f"Results written to {output_csv_filename}")
