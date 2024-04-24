#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE


vocab = open("data/brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    line = line.rstrip()
    word_index_dict[line] = i

f = open("data/brown_100.txt")

counts = np.zeros(len(word_index_dict))#TODO: initialize counts to a zero vector

#TODO: iterate through file and update counts
for i in f:
    words = i.lower().split()
    for word in words:
        if word in word_index_dict:
            index = word_index_dict[word]
            counts[index] += 1
print(counts) # print counts vector
f.close()

#TODO: normalize and writeout counts. 
probs = counts/np.sum(counts)
with open('result/unigram_probs.txt', 'w') as wf:
    wf.write(str(probs))

print(probs[0])
print(probs[-1])


