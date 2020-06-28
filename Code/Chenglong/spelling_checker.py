# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: spelling checker

"""

import gc
import time
import multiprocessing
from collections import Counter

import re, regex
import nltk
import difflib

import config
import google_spelling_checker_dict


class DictSpellingChecker:
    def __init__(self, spelling_checker_dict):
        self.spelling_checker_dict = spelling_checker_dict

    def correct(self, query):
        return self.spelling_checker_dict.get(query, query)


class GoogleQuerySpellingChecker(DictSpellingChecker):
    def __init__(self):
        super().__init__(google_spelling_checker_dict.spelling_checker_dict)


class PeterNovigSpellingChecker:
    """
    How to Write a Spelling Corrector
    http://norvig.com/spell-correct.html
    """
    def __init__(self, words, error_correction_pairs=None, exclude_stopwords=False, cutof