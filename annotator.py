import pandas as pd
import unidecode
import string
import re
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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

def decode_content(poem_data_row):
    poem_data_row["Content"] = unidecode.unidecode(poem_data_row["Content"])
    return poem_data_row

def main():
    print("Test:\n")
    annotator = Annotator()
    print(annotator.is_rhyme("sorry", "worry"))
    print(annotator.is_rhyme("vapid", "rapid"))
    print(annotator.is_rhyme("hey", "way"))
    print(annotator.is_rhyme("yo", "yoyo"))
    print(annotator.is_rhyme("love", "glove"))
    print(annotator.is_rhyme("punctual", "ethical"))

    poems = pd.read_csv("../data/poem_dataset_raw.csv").apply(decode_content, axis=1)
    poem2 = poems["Content"].get(2).split('\n')
    poem2_endline2 = poem2[2].split()[-1]
    poem2_endline3 = poem2[3].split()[-1]
    print(annotator.is_rhyme(poem2_endline2, poem2_endline3))

    poem2_rhyme_pairs = annotator.find_rhyme_pairs(poem2)
    print(poem2_rhyme_pairs)

    poems["Rhyme pairs"] = poems.apply(annotator.annotate_rhyme_pairs, axis=1)
    poems["POS tags"] = poems.apply(annotator.annotate_POS_tags, axis=1)
    # Filter out poems without rhyme pairs
    poems = poems[poems["Rhyme pairs"].apply(lambda x: len(x)) > 0]
    rhyme_pair_count = 0
    for index, row in poems.iterrows():
        content = row["Content"].split('\n')
        for rhyme_pair in row["Rhyme pairs"]:
            rhyme_pair_count += 1
            print("{} {}".format(content[rhyme_pair[0]].split()[-1], content[rhyme_pair[1]].split()[-1]))
    print("Found {} rhymes".format(rhyme_pair_count))

if __name__ == '__main__':
    main()