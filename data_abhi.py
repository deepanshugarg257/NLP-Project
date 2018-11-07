from collections import defaultdict

class Data(object):
    def __init__(self, name, restricted):
        self.name = name

        # All words in restricted list points to UNK.
        # Contents of list to be created by caller.
        self.restricted = restricted

        self.word2index = defaultdict(lambda: 3)
        self.word2index["PAD"] = 0
        self.word2index["SOS"] = 1
        self.word2index["EOS"] = 2
        self.word2index["UNK"] = 3
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.word2count[self.word2index["PAD"]] = 0
        self.word2count[self.word2index["SOS"]] = 0
        self.word2count[self.word2index["EOS"]] = 0
        self.word2count[self.word2index["UNK"]] = 0
        self.n_words = 4  # Count special tokens

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word in self.restricted:
            self.word2count[self.word2index["UNK"]] += 1
        elif word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[self.word2index[word]] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[self.word2index[word]] += 1