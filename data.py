import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.wordtoidx = {}
        self.idxtoword = []

    def add_word(self, word):
        if word not in self.wordtoidx:
            self.idxtoword.append(word)
            self.wordtoidx[word] = len(self.idxtoword) - 1
        return self.wordtoidx[word]

    def __len__(self):
        return len(self.idxtoword)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        print(f"Trying to access file: {path}")  
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.wordtoidx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
