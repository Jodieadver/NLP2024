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

### Trigram probabilities

trigrams_to_compute = [['past', 'in', 'the'], ['time','in', 'the'], ['said', 'the', 'jury'], ['reccomended', 'the', 'jury'], ['that','jury','said'], [',','agricolture', 'teacher']]

# the same above but all the trigrams in the correct order
trigrams_to_compute = [['in', 'the', 'past'], ['in', 'the', 'time'], ['the', 'jury', 'said'], ['the', 'jury', 'reccomended'], ['jury', 'said', 'that'], ['agricolture', 'teacher', ',']]


# we save the counts onlly for the trigrams we are interested in
trigram_counts = np.zeros((len(trigrams_to_compute), 2))

for line in f:
    words = line.lower().split()
    for i, word in enumerate(words[2:]):
        if  word not in word_index_dict:
            continue
        first_word = words[i]
        second_word = words[i+1]

        for i, [tri_first_word, tri_sec_word, tri_third_word] in enumerate(trigrams_to_compute):
            if tri_first_word == first_word and tri_sec_word == second_word:
                trigram_counts[i, 0] += 1 # count the number of anterior bigrams
                if tri_third_word == word:
                    trigram_counts[i, 1] += 1 # count the number of times the trigram appears
                    break

# initialize probabilities
probs = np.zeros(len(trigrams_to_compute))


for i, trigram_count in enumerate(trigram_counts):
    if trigram_count[0] == 0:
        probs[i] = 0
    else:
        probs[i] = trigram_count[1] / trigram_count[0]

# smoothing

trigram_counts[:, 0] += 0.1*len(word_index_dict)
trigram_counts[:, 1] += 0.1

probs_smooth = trigram_counts[:, 1] / trigram_counts[:, 0]

# writeout trigram probabilities

with open('result/trigram_probs.txt', 'w') as wf:
    for i, [tri_first_word, tri_sec_word, tri_third_word] in enumerate(trigrams_to_compute):
        prob = probs[i]
        prob_smooth = probs_smooth[i]
        wf.write(f'{tri_third_word}|{tri_first_word},{tri_sec_word}:\nprob: {prob} smooth prob:{prob_smooth}\n')


        

        
        
        

