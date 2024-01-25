import pandas as pd
import nltk
import string
from nltk.tag import pos_tag
import unidecode
from nltk.tokenize import word_tokenize
import re
import sys
import io
import csv
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
    
def decode_content(poem_data_row):
    poem_data_row["Content"] = unidecode.unidecode(poem_data_row["first_line"])
    return poem_data_row

def write_to_csv(data, file_name):
    field_names = ['Line', 'Groupnum']

    with open(file_name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        for key, value in data.items():
            try:
                writer.writerow({'Line': key, 'Groupnum': ','.join(map(str, value))})
            except UnicodeEncodeError:
                # Skip characters causing encoding issues
                cleaned_key = ''.join(c if ord(c) < 128 else '' for c in key)
                cleaned_value = [str(v) if isinstance(v, int) else v for v in value]
                writer.writerow({'Line': cleaned_key, 'Groupnum': ','.join(map(str, cleaned_value))})

def main():
    annotator = Annotator()
    print(annotator.is_rhyme("sorry", "worry"))
    print(annotator.is_rhyme("vapid", "rapid"))
    print(annotator.is_rhyme("hey", "way"))
    print(annotator.is_rhyme("yo", "yoyo"))
    print(annotator.is_rhyme("love", "glove"))
    print(annotator.is_rhyme("punctual", "ethical"))
    lyrics = pd.read_csv("./lyrics_rhyming_pairs.csv").apply(decode_content, axis=1)
    #print(lyrics)
    start_table=[]
    result=[]
    for i in range(len(lyrics)):
        start_table.append('unstart')
    print(len(lyrics))
    #lyrics2=lyrics["next_line"].get(2).split('\n')
    #print(lyrics2)
    #print(lyrics2[0].split())
    #group=dict()
    #groupnum=0
    for i in range(len(lyrics)):
        if start_table[i]=='unstart':
            lyrics1=lyrics['next_line'].get(i).split('n')
            last_word=lyrics1[0].split()
            try:
                last_word=last_word[-1]
            except: continue
            if len(result)==0:
                result.append([lyrics1[0]])
                #group[i]=[groupnum]
                #groupnum+=1
            else:
                for j in result:
                    last_word2=j[0].split()[-1]
                    if annotator.is_rhyme(last_word,last_word2):
                        j.append(lyrics1)
                        #group[i]=[group[result[j][0]][0]]
                        #print(group[result[j][0]])
                        #print(group[result[j][0]][0])
                        #print(result[j])
                    break
                result.append([lyrics1[0]])
                #group[i]=[groupnum]
                #groupnum+=1
            start_table[i]='start'
    #print(group)
    write_to_csv(result, 'output.csv')

    #df=pd.DataFrame(group)
    #df.to_csv('output.csv', index=False)
        

    #print(result['ma'])

        
    #print(lyrics2)
    #poem2_endline2 = poem2[2].split()[-1]
    #poem2_endline3 = poem2[3].split()[-1]
    #print(annotator.is_rhyme(poem2_endline2, poem2_endline3))


if __name__ == '__main__':
    main()