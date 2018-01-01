import os
import re
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def lem_count():
    if os.path.isfile('lemmatized.txt'):
        with open('lemmatized.txt') as l_dreams:
            return sum(1 for line in l_dreams) // 3
    else:
        return 0

def get_wordnet_pos(treebank_tag):
    #from: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN #wordnet default

def lemmatize(identifiers, dreams):
    l_count = lem_count() #number of entries already lemmatized

    if l_count != len(dreams):
        if l_count == 0:
            #write new lemmatized file
            mode = 'w'
        else:
            new_count = len(dreams) - l_count
            dreams = dreams[-new_count:]
            identifiers = identifiers[-new_count:]
            #add to existing lemmatized file
            mode = 'a'

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
                l_dreams.write(identifiers[i] + '\n')
                rem_delims = delim_pattern.sub(lambda m: delims[m.group(0)], dreams[i])
                tagged_dream = pos_tag(word_tokenize(rem_delims))
                for word in tagged_dream:
                    l_word = wnl.lemmatize((word[0][:-1].lower() + word[0][-1]), get_wordnet_pos(word[1]))
                    l_dreams.write(after_pattern.sub(lambda m: after_replace[m.group(0)], l_word))
                    l_dreams.write(' ')
                l_dreams.write("\n\n")
    print("Lemmatizing Done")
