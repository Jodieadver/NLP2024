
import sys
import os

class DualOutput:
    def __init__(self, filename, stdout):
        

        # check if the file exists
        if os.path.exists(filename):
            os.remove(filename)
        # create directory and file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.file = open(filename, "w")
        self.stdout = stdout
        



    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):  # Needed for compatibility with file flushing
        self.file.flush()
        self.stdout.flush()

    def close(self):  # Not usually necessary, but a good practice to define
        self.file.close()

# Replace sys.stdout with an instance of DualOutput
sys.stdout = DualOutput("result/problem0.txt", sys.stdout)


# %% [markdown]
# # Excercise 0 (Bag of words)

# %% [markdown]
# ## Ordering of words frequences

# %% [markdown]
# ### Whole Brow corpus

# %% [markdown]
# Computing a list of unique words sorted by descending frequency for the whole corpus

# %%
# import brown corpus from NLTK
from nltk.corpus import brown
import nltk

words = brown.words()

# remove punctuation from the tokena
words = [word for word in words if word.isalpha()]

# get the frequency distribution of all words
fdist = nltk.FreqDist(words)

# sort the frequency distribution by descending frequency
sorted_fdist = sorted(fdist.items(), key=lambda x: x[1], reverse=True)


print("\n-- Common words --")

# print the 10 most common words
print("10 most common words in the whole corpus:")
for i in range(10):
    print(sorted_fdist[i])


# %% [markdown]
# ### Two specific genres

# %% [markdown]
# Computing a list of unique words sorted by descending frequency for two different genres of our choice

# %%

#---NEWS GENRE---

news_words = brown.words(categories='news')

# Remove punctuation from the words
news_words = [word for word in news_words if word.isalpha()]

news_fdist = nltk.FreqDist(news_words)

# sort the frequency distribution by descending frequency
sorted_news_fdist = sorted(news_fdist.items(), key=lambda x: x[1], reverse=True)


print("\n10 most common words in the news genre:")

for i in range(10):
    print(sorted_news_fdist[i])

#---ROMANCE GENRE---

romance_words = brown.words(categories='romance')

romance_words = [word for word in romance_words if word.isalpha()]

romance_fdist = nltk.FreqDist(romance_words)

sorted_romance_fdist = sorted(romance_fdist.items(), key=lambda x: x[1], reverse=True)

print("\n10 most common words in the romance genre:")
for i in range(10):
    print(sorted_romance_fdist[i])

# %% [markdown]
# ## Extracting number of tokens, types, words etc.

# %% [markdown]
# Extracting the following information: number of tokens; number of types; number of words; average number of words per sentence; average word length

# %%
def extract_info(tokens, sentences):

    # number of tokens
    num_tokens = len(tokens)

    # number of types
    fdist = nltk.FreqDist(tokens)
    num_types = len(fdist)

    # number of words
    words = [word for word in tokens if word.isalpha()]
    num_words = len(words)

    # average number of words per sentence
    num_sentences = len(sentences)
    avg_words_per_sentence = num_words / num_sentences

    # average word length
    total_word_length = sum([len(word) for word in words])
    avg_word_length = total_word_length / num_words

    return num_tokens, num_types, num_words, avg_words_per_sentence, avg_word_length

# %%


print("\n-- Statistics for different corpora --\n")
num_tokens, num_types, num_words, avg_words_per_sentence, avg_word_length = extract_info(brown.words(), brown.sents())

print("\nBrown corpus:")
print(f"Number of tokens (including punct): {num_tokens}\nNumber of words: {num_words}\nNumber of distinct words (types): {num_types}\nAverage number of words per sentence: {avg_words_per_sentence:.2f}\nAverage word length: {avg_word_length:.2f}")

num_tokens_news, num_types_news, num_words_news, avg_words_per_sentence_news, avg_word_length_news = extract_info(brown.words(categories='news'), brown.sents(categories='news'))

print("\nNews genre:")
print(f"Number of tokens (including punct): {num_tokens_news}\nNumber of words: {num_words_news}\nNumber of distinct words (types): {num_types_news}\nAverage number of words per sentence: {avg_words_per_sentence_news:.2f}\nAverage word length: {avg_word_length_news:.2f}")

num_tokens_romance, num_types_romance, num_words_romance, avg_words_per_sentence_romance, avg_word_length_romance = extract_info(brown.words(categories='romance'), brown.sents(categories='romance'))

print("\nRomance genre:")
print(f"Number of tokens (including punct): {num_tokens_romance}\nNumber of words: {num_words_romance}\nNumber of distinct words (types): {num_types_romance}\nAverage number of words per sentence: {avg_words_per_sentence_romance:.2f}\nAverage word length: {avg_word_length_romance:.2f}")

# %% [markdown]
# ## Part of speech tagging

# %% [markdown]
# Running a default part-of-speech tagger on the dataset and identify the ten most frequent POS tags.

# %%
#function to  run a default part-of-speech tagger on the dataset and identify the ten most frequent POS tags.

def pos_tags(tokens):
    # remove punctuation from the tokens
    tokens = [token for token in tokens if token.isalpha()]

    # run the default part-of-speech tagger on the dataset
    tagged = nltk.pos_tag(tokens)



    # get the frequency distribution of the tags
    fdist = nltk.FreqDist(tag for (word, tag) in tagged)

    # sort the frequency distribution by descending frequency
    sorted_fdist = sorted(fdist.items(), key=lambda x: x[1], reverse=True)

    return sorted_fdist

print("\n-- Most frequent Part of speech tags --")

# %%
print("\nBrown corpus:")

sorted_fdist = pos_tags(brown.words())
for i in range(10):
    print(sorted_fdist[i])

print("\nNews genre:")
sorted_fdist_news = pos_tags(brown.words(categories='news'))
for i in range(10):
    print(sorted_fdist_news[i])

print("\nRomance genre:")
sorted_fdist_romance = pos_tags(brown.words(categories='romance'))
for i in range(10):
    print(sorted_fdist_romance[i])

# %% [markdown]
# ## Plotting frequences

# %% [markdown]
# Using the Python library matplotlib to plot the frequency curves for the corpus and two genres you choose: i.e. x-axis is position in the frequency list, y-axis is frequency. Provide both a plot with linear axes and one with log-log axes.

# %%
import matplotlib.pyplot as plt

# plot the frequency curves for the corpus and two genres
def plot_frequency_curve(tokens, title):

    tokens = [token for token in tokens if token.isalpha()]
    fdist = nltk.FreqDist(tokens)
    sorted_fdist = sorted(fdist.items(), key=lambda x: x[1], reverse=True)
    
    x = range(len(sorted_fdist))
    y = [freq for (word, freq) in sorted_fdist]

    # make two plots: one with linear axes and one with log-log axes
    fig, axs = plt.subplots(2)
    fig.suptitle(title)
    axs[0].plot(x, y)
    axs[0].set_title("Linear axes")

    axs[1].plot(x, y)
    axs[1].set_title("Log-log axes")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    plt.tight_layout()

    plt.show()
    


# %%
plot_frequency_curve(brown.words(), 'Brown corpus')
plot_frequency_curve(brown.words(categories='news'), 'News genre')
plot_frequency_curve(brown.words(categories='romance'), 'Romance genre')

# %% [markdown]
# 


