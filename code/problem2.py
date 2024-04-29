#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE

# problem 2
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
    for word, prob in zip(word_index_dict.keys(), probs):
        wf.write(f"{word}|{prob}\n")

print(f"Probability of the word '{list(word_index_dict.keys())[0]}': {probs[0]}")
print(f"Probability of the word '{list(word_index_dict.keys())[-1]}': {probs[-1]}")



# problem 6
# read the unigram_probs file
with open('result/unigram_probs.txt', 'r') as f:
    unigram_probs = {}
    for line in f:
        word, probability = line.strip().split('|')
        unigram_probs[word] = float(probability)
f.close()
        
f_toy = open("data/toy_corpus.txt")

#TODO: iterate through file and update counts
with open('result/unigram_eval.txt', 'w') as wf:
    for i in f_toy:
        words = i.lower().split()
        sentprob = 1
        for word in words:
            wordprob = unigram_probs[word]
            sentprob *= wordprob
        
        sen_len = len(words) 
        perplexity = 1 / (sentprob ** (1.0 / sen_len))
        wf.write(f"Perplexity : {perplexity}\n")

f_toy.close()
