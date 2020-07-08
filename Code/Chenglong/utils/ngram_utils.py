# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for ngram

"""


def _unigrams(words):
    """
        Input: a list of words, e.g., ["I", "am", "Denny"]
        Output: a list of unigram
    """
    assert type(words) == list
    return words


def _bigrams(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ["I", "am", "Denny"]
       Output: a list of bigram, e.g., ["I_am", "am_Denny"]
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for k in range(1,skip+2):
                if i+k < L:
                    lst.append( join_string.join([words[i], words[i+k]]) )
    else:
        # set it as unigram
        lst = _unigrams(words)
    return lst


def _trigrams(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ["I", "am", "Denny"]
       Output: a list of trigram, e.g., ["I_am_Denny"]
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in range(L-2):
            for k1 in range(1,skip+2):
                for k2 in range(1,skip+2):
                    if i+k1 < L and i+k1+k2 < L:
                        lst.append( join_string.join([words[i], words[i+k1], words[i+k1+k2]]) )
    else:
        # set it as bigram
        lst = _bigrams(words, join_string, skip)
    return lst


def _fourgrams(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of trigram, e.g., ["I_am_Denny_boy"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 3:
        lst = []
        for i in xrange(L-3):
            lst.append( join_string.join([words[i], words[i+1], words[i+2], words[i+3]]) )
    else:
        # set it as trigram
        lst = _trigrams(words, join_string)
    return lst


def _uniterms(words):
    return _unigrams(words)


def _biterms(