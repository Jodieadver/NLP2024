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
counts = np.zeros((len(word_index_dict), len(word_index_dict))) #TODO: initialize numpy 0s array

#TODO: iterate through file and update counts
previous_word = '<s>'
for line in f:
    words = line.lower().split()
    for word in words[1:]: # skip the first word
        if word in word_index_dict:
            counts[word_index_dict[word], word_index_dict[previous_word]] += 1
            previous_word = word

print(f" counts of 'all' :{counts[:, word_index_dict['all']].sum()}")
print(f" counts of 'all the' :{counts[word_index_dict['the'], word_index_dict['all']]}")
      
#TODO: normalize counts
probs = normalize(counts, norm='l1', axis=0)

# print the probability of the bigram 'all the'
print(f"probability of 'all the': {probs[word_index_dict['the'], word_index_dict['all']]}")

#TODO: writeout bigram probabilities

pairs_to_test = [('the','all'), ('jury', 'the'), ('campaign', 'the'), ('calls', 'anonymous')]

with open('result/bigram_probs.txt', 'w') as wf:
    for pair in pairs_to_test:
        prob = probs[word_index_dict[pair[0]], word_index_dict[pair[1]]]
        wf.write(f'{pair[0]}|{pair[1]}: {prob}\n')


f.close()

# test
#prob1 = probs[word_index_dict['all'], word_index_dict['the']]
#prob2 = probs[word_index_dict['jury'], word_index_dict['the']]
#prob3 = probs[word_index_dict['campaign'], word_index_dict['the']]
#prob4 = probs[word_index_dict['calls'], word_index_dict['anonymous']]
#print('the|all:', prob1)
#print('jury|the:', prob2)