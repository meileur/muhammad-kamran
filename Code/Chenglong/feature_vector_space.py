# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: vector space based features
        - char/word based ngram LSA & cosine similarity
        - char/word based ngram TFIDF & cosine similarity
        - cooccurrence & pairwise LSA
        - standalone and pairwise TSNE (memory error, use feature_tsne.R instead)
        - char distribution based cosine similarity and KL divergence

"""

import os
import re
import string

import scipy
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

try:
    from tsne import bh_sne
except:
    pass

import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils, pkl_utils
from utils import logging_utils, time_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper, PairwiseFeatureWrapper


class VectorSpace:

    ## word based
    def _init_word_bow(self, ngram, vocabulary=None):
        bow = CountVectorizer(min_df=3,
                                max_df=0.75,
                                max_features=None,
                                # norm="l2",
                                strip_accents="unicode",
                                analyzer="word",
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, ngram),
                                vocabulary=vocabulary)
        return bow

    ## word based
    def _init_word_ngram_tfidf(self, ngram, vocabulary=None):
        tfidf = TfidfVectorizer(min_df=3,
                                max_df=0.75,                                
                                max_features=None,
                                norm="l2",
                                strip_accents="unicode",
                                analyzer="word",
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, ngram),
                                use_idf=1,
                                smooth_idf=1,
                                sublinear_tf=1,
                                # stop_words="english",
                                vocabulary=vocabulary)
        return tfidf

    ## char based
    def _init_char_tfidf(self, include_digit=False):
        chars = list(string.ascii_lowercase)
        if include_digit:
            chars += list(string.digits)        
        vocabulary = dict(zip(chars, range(len(chars))))
        tfidf = TfidfVectorizer(strip_accents="unicode",
                                analyzer="char",
                                norm=None,
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, 1), 
                                use_idf=0,
                                vocabulary=vocabulary)
        return tfidf

    ## char based ngram
    def _init_char_ngram_tfidf(self, ngram, vocabulary=None):
        tfidf = TfidfVectorizer(min_df=3,
                                max_df=0.75, 
                                max_features=None, 
                                norm="l2",
                                strip_accents="unicode", 
                                analyzer="char",
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, ngram), 
                                use_idf=1,
                                smooth_idf=1,
                                sublinear_tf=1, 
                                # stop_words="english",
                                vocabulary=vocabulary)
        return tfidf


# ------------------------ LSA -------------------------------
class LSA_Word_Ngram(VectorSpace):
    def __init__(self, obs_corpus, place_holder, ngram=3, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        
    def __name__(self):
        return "LSA%d_Word_%s"%(self.svd_dim, self.ngram_str)

    def transform(self):
        tfidf = self._init_word_ngram_tfidf(self.ngram)
        X = tfidf.fit_transform(self.obs_corpus)
        svd = TruncatedSVD(n_components = self.svd_dim, 
                n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        return svd.fit_transform(X)


class LSA_Char_Ngram(VectorSpace):
    def __init__(self, obs_corpus, place_holder, ngram=5, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        
    def __name__(self):
        return "LSA%d_Char_%s"%(self.svd_dim, self.ngram_str)

    def transform(self):
        tfidf = self._init_char_ngram_tfidf(self.ngram)
        X = tfidf.fit_transform(self.obs_corpus)
        svd = TruncatedSVD(n_components=self.svd_dim, 
                n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        return svd.fit_transform(X)


# ------------------------ Cooccurrence LSA -------------------------------
# 1st in CrowdFlower
class LSA_Word_Ngram_Cooc(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, 
            obs_ngram=1, target_ngram=1, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.obs_ngram = obs_ngram
        self.target_ngram = target_ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.obs_ngram_str = ngram_utils._ngram_str