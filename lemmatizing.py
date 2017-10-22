import os
import re
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

dreams = []
identifiers = []

with open('dreams.txt', encoding='ascii', errors = 'ignore') as dreamstxt:
    for i, line in enumerate(dreamstxt):
        if i % 3 == 0:
            identifiers.append(line)
        elif (i - 1) % 3 == 0:
            dreams.append(line)

l_count = 0 #number of entries already lemmatized

if os.path.isfile('lemmatized.txt'):
    #add to existing lemmatized file
    mode = 'a'
    with open('lemmatized.txt') as l_dreams:
        l_count = sum(1 for line in l_dreams) // 3
    new_count = len(dreams) - l_count
    dreams = dreams[-new_count:]
    identifiers = identifiers[-new_count:]
else:
    #write new lemmatized file
    mode = 'w'

#from: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN #wordnet default

if l_count != len(dreams):
    #from: https://emilics.com/notebook/enblog/p869.html
    delims = {
        #remove some separating characters without breaking ability
        #to accurately separate contractions using a tokenizer
        "/": " ",
        "-": " "
    }
    delim_pattern = re.compile("|".join(delims.keys()))
    after_replace = {
        #replace separated pieces of contractions with full words
        "n't": "not",
        "'re": "are",
        "'ve": "have",
        "'ll": "will",
        #replace numbers with words because single characters are
        #ignored by default in scikit-learn's CountVectorizer
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine"
    }
    after_pattern = re.compile("|".join(after_replace.keys()))

    wnl = WordNetLemmatizer()
    with open('lemmatized.txt', mode) as l_dreams:
        for i in range(len(dreams)):
            l_dreams.write(identifiers[i])
            rem_delims = delim_pattern.sub(lambda m: delims[m.group(0)], dreams[i])
            tagged_dream = pos_tag(word_tokenize(rem_delims))
            for word in tagged_dream:
                l_word = wnl.lemmatize((word[0][:-1].lower() + word[0][-1]), get_wordnet_pos(word[1]))
                l_dreams.write(after_pattern.sub(lambda m: after_replace[m.group(0)], l_word))
                l_dreams.write(' ')
            l_dreams.write("\n\n")
print("Lemmatizing Done")
