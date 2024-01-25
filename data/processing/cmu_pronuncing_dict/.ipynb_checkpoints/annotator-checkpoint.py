import pandas as pd
import pronouncing
import os
import unidecode

class Annotator:
    def __init__(self):
        self.__phoneme_table = self.__create_phoneme_table()

    def __create_phoneme_table(self):
        # the phoneme table is a pandas DataFrame of columns "words" and
        # "phonemes" the phoneme being the list of phonemes corresponding
        # to that word in the CMU Pronouncing Dictionary
        table = dict()

        with open("data/processing/cmu_pronuncing_dict/cmudict-0.7b", 'r', encoding="iso8859_2") as cmu_dict_file:
            for i in range(126):
                next(cmu_dict_file)
            while True:
                try:
                    line = next(cmu_dict_file).split()
                except StopIteration:
                    break
                table[line[0]] = line[1:]

        return table
    
    # expects lyrics as a list of lines of words,
    # the first index indexes the line
    # the second index indexes the word in that line
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
    
    def get_phoneme(self, word):
        return self.__phoneme_table[word.upper()]
    
    # finds the last syllable in the word with primary or secondary stress,
    # and returns true if this syllable and the ones following it match
    # those in the otherword
    def is_rhyme(self, word, otherword):
        try:
            wordphonemes = self.get_phoneme(word)
            otherwordphonemes = self.get_phoneme(otherword)
        except KeyError: return False
        ind = 0
        for i in reversed(range(len(wordphonemes))):
            # checks if the phoneme has character 1 or 2 at the end, i.e.
            # if it's a vowel with primary or secondary stress
            if wordphonemes[i][-1] in ['1','2']:
                ind = i
                break
        '''while ind > 0:
            if wordphonemes[ind-1][-1] not in ['0','1','2']:
                ind -= 1
            else: break'''
        rhymephonemes = wordphonemes[ind:]
        return otherwordphonemes[-len(rhymephonemes):] == rhymephonemes
                        
def main():
    print("Test:\n")
    annotator = Annotator()
    print(annotator.is_rhyme("sorry", "worry"))
    print(annotator.is_rhyme("vapid", "rapid"))
    print(annotator.is_rhyme("hey", "way"))
    print(annotator.is_rhyme("yo", "yoyo"))
    print(annotator.is_rhyme("love", "glove"))
    print(annotator.is_rhyme("punctual", "ethical"))

    poems = pd.read_csv("data/poem_dataset_raw.csv")
    poem2 = unidecode.unidecode(poems["Content"].get(2)).split('\n')
    poem2_endline2 = poem2[2].split()[-1]
    poem2_endline3 = poem2[3].split()[-1]
    print(annotator.is_rhyme(poem2_endline2, poem2_endline3))

    poem2_rhyme_pairs = annotator.find_rhyme_pairs(poem2)
    print(poem2_rhyme_pairs)

if __name__ == '__main__':
    main()