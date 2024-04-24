#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

word_index_dict = {}

# TODO: read brown_vocab_100.txt into word_index_dict
with open('data/brown_vocab_100.txt', 'r') as file:
    for index, word in enumerate(file):
            word = word.rstrip() #remove newline character
            word_index_dict[word] = index 

# TODO: write word_index_dict to word_to_index_100.txt
with open('word_to_index_100.txt', 'w') as wf:
      wf.write(str(word_index_dict))


print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
