#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random


# vocab = codecs.open("data/brown_vocab_100.txt") 
vocab = open("data/brown_vocab_100.txt") 

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    line = line.rstrip()
    word_index_dict[line] = i


#f = codecs.open("data/brown_100.txt")
f = open("data/brown_100.txt")
# first parameter is to count how many lines the txt file have, second is to count 
bigram_counts = np.zeros((len(word_index_dict), len(word_index_dict))) #initialize numpy 0s array
unigram_counts = np.zeros(len(word_index_dict))

#iterate through file and update counts
previous_word = '<s>'
for line in f:
    words = line.lower().split()
    for word in words[1:]: # skip the first word
        if word in word_index_dict:
            unigram_counts[word_index_dict[word]] += 1
            bigram_counts[word_index_dict[previous_word], word_index_dict[word]] += 1
            previous_word = word



print(f" counts of 'all' :{bigram_counts[word_index_dict['all'], :].sum()}")
print(f" counts of 'all the' :{bigram_counts[word_index_dict['all'], word_index_dict['the']]}")
      
# alpha smoothing
smoothed_bigram_counts = bigram_counts + 0.1
smoothed_unigram_counts = unigram_counts + 0.1

#normalize counts
bigram_probs = normalize(bigram_counts, norm='l1', axis=1)
smoothed_probs = normalize(smoothed_bigram_counts, norm='l1', axis=1)

unigram_probs = unigram_counts/unigram_counts.sum()
smoothed_unigram_probs = smoothed_unigram_counts/smoothed_unigram_counts.sum()

# print the probability of the bigram 'all the'
print(f"probability of 'all the'--\nNormal: {bigram_probs[word_index_dict['all'], word_index_dict['the']]} Smoothed: {smoothed_probs[word_index_dict['all'], word_index_dict['the']]}")


# Generate senteces and save them to a file

n_of_sentences = 10

with open('result/unigram_generation.txt', 'w') as wf:
    for i in range(n_of_sentences):
        sentence = GENERATE(word_index_dict, unigram_probs, 'unigram', 100, '<s>')
        wf.write(f'{sentence}\n')

with open('result/bigram_generation.txt', 'w') as wf:
    for i in range(n_of_sentences):
        sentence = GENERATE(word_index_dict, bigram_probs, 'bigram', 100, '<s>')
        wf.write(f'{sentence}\n')

with open('result/smoothed_generation.txt', 'w') as wf:
    for i in range(n_of_sentences):
        sentence = GENERATE(word_index_dict, smoothed_probs, 'bigram', 100, '<s>')
        wf.write(f'{sentence}\n')
