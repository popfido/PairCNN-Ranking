# coding=utf-8

import os
import codecs
from torch.utils import data
import itertools
from functools import reduce
import numpy as np
from torchvision import transforms as T
from collections import Counter


def build_vocab(files, vocab_size, validation_func=lambda x: len(x) == 3):
    # Build vocabulary
    word_counts = Counter()
    valid_lines_num = []
    for file in files:
        valid_lines_counter = 0
        f = codecs.open(file, 'r')
        for line in f:
            data = line.strip().split(',')
            if validation_func(data):
                valid_lines_counter += 1
                word_counts.update(reduce(lambda x, y: x + y, map(lambda x: x.strip().split(' '), data[:2])))
            else:
                print("There exists a line whose format not match the validation criteria: " + line)
        f.close()
        if len(valid_lines_num) == 0:
            valid_lines_num.append(valid_lines_counter)
        else:
            valid_lines_num.append(valid_lines_counter + valid_lines_num[-1])
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size - 1)]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary_inv.append('<UNK/>')
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv, valid_lines_num]


def build_validation(files, vocab, validation_func=lambda x: len(x) == 3):
    valid_lines_num = []
    for file in files:
        valid_lines_counter = 0
        f = codecs.open(file, 'r')
        for line in f:
            data = line.strip().split(',')
            if validation_func(data):
                if list(filter(lambda x: x in vocab,
                               reduce(lambda x, y: x + y, map(lambda x: x.strip().split(' '), data[:2])))):
                    valid_lines_counter += 1
            else:
                print("There exists a line whose format not match the validation criteria: " + line)
        f.close()
        if len(valid_lines_num) == 0:
            valid_lines_num.append(valid_lines_counter)
        else:
            valid_lines_num.append(valid_lines_counter + valid_lines_num[-1])
    return valid_lines_num


class DocumentPair(data.Dataset):
    def __init__(self, root, suffix, vocab=None, doc_type='Train', load=lambda x: x):
        """
        主要目标： 获取所有文件的地址
        """
        super(DocumentPair, self).__init__()

        self.load = load
        self.vocab = vocab
        self.vocab_inv = None
        self.files = [os.path.join(root, file) for file in os.listdir(root) if suffix in file]

        self.type = doc_type
        self.validation_func = lambda x: len(x) == 3
        self.data = None
        self.num_data = []
        self.now_file_index = 0
        self.now_line_index = 0
        self.file = None
        self.initialized = False

    def initialize(self, vocab_size=None):
        if self.type == "train":
            vocab, vocab_inv, self.num_data = build_vocab(self.files, vocab_size, self.validation_func)
            self.vocab = vocab
            self.vocab_inv = vocab_inv
        elif self.type == "validate":
            if self.vocab is None:
                raise ValueError("Vocabulary should be provided to Validation Set")
            self.num_data = build_validation(self.files, self.vocab, self.validation_func)
        elif self.type == "test":
            if self.vocab is None:
                raise ValueError("Vocabulary should be provided to Test Set")
        else:
            raise ValueError("Type of dataset is not in [train, validate, test]")

    def __getitem__(self, index):
        """
        一次返回一个位置的文本样本
        """
        if index < 0 or index >= self.num_data[-1]:
            raise IndexError("input index is out of bound")

        in_file_index = 0
        in_line_index = index
        for i in range(len(self.num_data)):
            if in_line_index < self.num_data[i]:
                in_file_index = i
                break

        if self.file is None or in_file_index != self.now_file_index:
            f = codecs.open(self.files[in_file_index])
            self.data = list(filter(self.validation_func, [self.load(line) for line in f]))
            f.close()
        return self.data[in_line_index - self.num_data[in_file_index]]

    def __len__(self):
        return self.num_data[-1]


if __name__ == '__main__':
    docs = DocumentPair(root='../data/train', vocab=None, doc_type='train',
                        load=lambda x: x.strip().split(','), suffix='txt')
    docs.initialize(100000)
    print(len(docs))
    loader = data.DataLoader(docs, batch_size=128)
    for ii, batch in enumerate(loader):
        print(ii)
    val_data = DocumentPair(root='../data/validate', doc_type='validate',
                            suffix='txt', load=lambda x: x.strip().split(','), vocab=docs.vocab)
    val_data.initialize()
    print(len(val_data))
    loader = data.DataLoader(val_data, batch_size=128)
    for ii, batch in enumerate(loader):
        print(ii)