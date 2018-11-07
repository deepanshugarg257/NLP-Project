from io import open
import unicodedata
import re
import pickle

from data import Data

class Data_Preprocess(object):
    def __init__(self, max_length, v1, v2):
        self.max_length = max_length

    def read_langs(self, reverse=False):
        print("Reading lines...")

        # Read the file and split into lines
        eng_lines = open('./model/dataset/en-hi.en', encoding='utf-8').readlines()

        # Split every line into pairs and normalize
        eng = [self.normalize_string(l).split(' ') for l in eng_lines]

        with open("./model/dataset/hindi_tokens.txt", "rb") as fp:
            hin = pickle.load(fp)

        ####### tokenization done
        print("eng lines read", len(eng))
        print("hin lines read", len(hin))

        # for i in range(5):
        #     print(hin[i], eng[i])

        # Reverse pairs, make Data instances
        if reverse:
            pairs = list(zip(eng, hin))
            input_lang = Data('eng')
            output_lang = Data('hin')
        else:
            pairs = list(zip(hin, eng))
            input_lang = Data('hin')
            output_lang = Data('eng')

        return input_lang, output_lang, pairs

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def filter_pair(self, p):
        return len(p[0]) < self.max_length and \
               len(p[1]) < self.max_length

    def filter_pairs(self, pairs):
        return [pair for count, pair in enumerate(pairs) if self.filter_pair(pair)]

    def prepare_data(self, reverse=False):
        input_lang, output_lang, pairs = self.read_langs(reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = self.filter_pairs(pairs)

        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for i, pair in enumerate(pairs):
            if i%10000 == 0:
                print(i, " sentences added of ", len(pairs))
                with open("out.log", "a") as myfile:
                    myfile.write(str(i) + " sentences added of " + str(len(pairs)) + "\n")
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])

        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)

        # Sort data in reverse order of lengths for easy batch processing
        pairs = sorted(pairs, key=lambda l: len(l[0]), reverse=True)
        return input_lang, output_lang, pairs
