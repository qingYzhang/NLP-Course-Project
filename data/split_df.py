import pandas as pd
import numpy as np

import re
import random

import nltk
import string
from nltk.tag import pos_tag
import unidecode
from nltk.tokenize import word_tokenize
import re
import sys
import io
import csv
import tensorflow as tf
import tensorflow
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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

        with open("./processing/cmu_pronuncing_dict/cmudict-0.7b", 'r', encoding="iso8859_2") as cmu_dict_file:
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

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


## Change path to destination on your end
path = './lyrics_rhyming_pairs.csv'

def ngram(token_list):
  ng = []
  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    ng.append(n_gram_sequence)
  return ng

df = pd.read_csv(path)
df['last_word'] = df.next_line.str.split().str[-1]
df['label'] = 0


# Iterate through the DataFrame
# Iterate through the DataFrame
i = 1
for index, row in df.iterrows():
    target_word = row['last_word']

    # Skip if the target word has already been labeled
    if row['label'] != 0:
      continue
    df.iloc[index, 4] = i
    
    # Iterate through subsequent words
    for sub_index in range(index + 1, len(df)):
        if df.iloc[sub_index, 4] != 0:
          continue
        sub_word = df.at[sub_index, 'last_word']
        # Check if the word rhymes and hasn't been labeled
        annotator = Annotator()
        if annotator.is_rhyme(target_word, sub_word) :
            df.at[sub_index, 'label'] = i
    i += 1
    
df.to_csv("new_grouped_lyrics.csv", index=False)
