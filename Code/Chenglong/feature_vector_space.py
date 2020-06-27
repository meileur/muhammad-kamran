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
        self.obs_ngram_str = ngram_utils._ngram_str_map[self.obs_ngram]
        self.target_ngram_str = ngram_utils._ngram_str_map[self.target_ngram]

    def __name__(self):
        return "LSA%d_Word_Obs_%s_Target_%s_Cooc"%(
            self.svd_dim, self.obs_ngram_str, self.target_ngram_str)

    def _get_cooc_terms(self, lst1, lst2, join_str):
        out = [""] * len(lst1) * len(lst2)
        cnt =  0
        for item1 in lst1:
            for item2 in lst2:
                out[cnt] = item1 + join_str + item2
                cnt += 1
        res = " ".join(out)
        return res

    def transform(self):
        # ngrams
        obs_ngrams = list(map(lambda x: ngram_utils._ngrams(x.split(" "), self.obs_ngram, "_"), self.obs_corpus))
        target_ngrams = list(map(lambda x: ngram_utils._ngrams(x.split(" "), self.target_ngram, "_"), self.target_corpus))
        # cooccurrence ngrams
        cooc_terms = list(map(lambda lst1,lst2: self._get_cooc_terms(lst1, lst2, "X"), obs_ngrams, target_ngrams))
        ## tfidf
        tfidf = self._init_word_ngram_tfidf(ngram=1)
        X = tfidf.fit_transform(cooc_terms)
        ## svd
        svd = TruncatedSVD(n_components=self.svd_dim, 
                n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        return svd.fit_transform(X)


# 2nd in CrowdFlower (preprocessing_mikhail.py)
class LSA_Word_Ngram_Pair(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=2, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "LSA%d_Word_%s_Pair"%(self.svd_dim, self.ngram_str)

    def transform(self):
        ## tfidf
        tfidf = self._init_word_ngram_tfidf(ngram=self.ngram)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        X_target = tfidf.fit_transform(self.target_corpus)
        X_tfidf = scipy.sparse.hstack([X_obs, X_target]).tocsr()
        ## svd
        svd = TruncatedSVD(n_components=self.svd_dim, 
                n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        X_svd = svd.fit_transform(X_tfidf)
        return X_svd


# -------------------------------- TSNE ------------------------------------------
# 2nd in CrowdFlower (preprocessing_mikhail.py)
class TSNE_LSA_Word_Ngram(LSA_Word_Ngram):
    def __init__(self, obs_corpus, place_holder, ngram=3, svd_dim=100, svd_n_iter=5):
        super().__init__(obs_corpus, None, ngram, svd_dim, svd_n_iter)
        
    def __name__(self):
        return "TSNE_LSA%d_Word_%s"%(self.svd_dim, self.ngram_str)

    def transform(self):
        X_svd = super().transform()
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = TSNE().fit_transform(X_scaled)
        return X_tsne


class TSNE_LSA_Char_Ngram(LSA_Char_Ngram):
    def __init__(self, obs_corpus, place_holder, ngram=5, svd_dim=100, svd_n_iter=5):
        super().__init__(obs_corpus, None, ngram, svd_dim, svd_n_iter)
        
    def __name__(self):
        return "TSNE_LSA%d_Char_%s"%(self.svd_dim, self.ngram_str)

    def transform(self):
        X_svd = super().transform()
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = TSNE().fit_transform(X_scaled)
        return X_tsne


class TSNE_LSA_Word_Ngram_Pair(LSA_Word_Ngram_Pair):
    def __init__(self, obs_corpus, target_corpus, ngram=2, svd_dim=100, svd_n_iter=5):
        super().__init__(obs_corpus, target_corpus, ngram, svd_dim, svd_n_iter)

    def __name__(self):
        return "TSNE_LSA%d_Word_%s_Pair"%(self.svd_dim, self.ngram_str)

    def transform(self):
        X_svd = super().transform()
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = TSNE().fit_transform(X_scaled)
        return X_tsne


# ------------------------ TFIDF Cosine Similarity -------------------------------
class TFIDF_Word_Ngram_CosineSim(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=3):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        
    def __name__(self):
        return "TFIDF_Word_%s_CosineSim"%self.ngram_str

    def transform(self):        
        ## get common vocabulary
        tfidf = self._init_word_ngram_tfidf(self.ngram)
        tfidf.fit(list(self.obs_corpus) + list(self.target_corpus))
        vocabulary = tfidf.vocabulary_
        ## obs tfidf
        tfidf = self._init_word_ngram_tfidf(self.ngram, vocabulary)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        ## targetument tfidf
        tfidf = self._init_word_ngram_tfidf(self.ngram, vocabulary)
        X_target = tfidf.fit_transform(self.target_corpus)
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


class TFIDF_Char_Ngram_CosineSim(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "TFIDF_Char_%s_CosineSim"%self.ngram_str

    def transform(self):
        ## get common vocabulary
        tfidf = self._init_char_ngram_tfidf(self.ngram)
        tfidf.fit(list(self.obs_corpus) + list(self.target_corpus))
        vocabulary = tfidf.vocabulary_
        ## obs tfidf
        tfidf = self._init_char_ngram_tfidf(self.ngram, vocabulary)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        ## targetument tfidf
        tfidf = self._init_char_ngram_tfidf(self.ngram, vocabulary)
        X_target = tfidf.fit_transform(self.target_corpus)
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


# ------------------------ LSA Cosine Similarity -------------------------------
class LSA_Word_Ngram_CosineSim(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=3, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "LSA%d_Word_%s_CosineSim"%(self.svd_dim, self.ngram_str)

    def transform(self):
        ## get common vocabulary
        tfidf = self._init_word_ngram_tfidf(self.ngram)
        tfidf.fit(list(self.obs_corpus) + list(self.target_corpus))
        vocabulary = tfidf.vocabulary_
        ## obs tfidf
        tfidf = self._init_word_ngram_tfidf(self.ngram, vocabulary)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        ## targetument tfidf
        tfidf = self._init_word_ngram_tfidf(self.ngram, vocabulary)
        X_target = tfidf.fit_transform(self.target_corpus)
        ## svd
        svd = TruncatedSVD(n_components = self.svd_dim, 
                n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        svd.fit(scipy.sparse.vstack((X_obs, X_target)))
        X_obs = svd.transform(X_obs)
        X_target = svd.transform(X_target)
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


class LSA_Char_Ngram_CosineSim(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=5, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        
    def __name__(self):
        return "LSA%d_Char_%s_CosineSim"%(self.svd_dim, self.ngram_str)

    def transform(self):
        ## get common vocabulary
        tfidf = self._init_char_ngram_tfidf(self.ngram)
        tfidf.fit(list(self.obs_corpus) + list(self.target_corpus))
        vocabulary = tfidf.vocabulary_
        ## obs tfidf
        tfidf = self._init_char_ngram_tfidf(self.ngram, vocabulary)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        ## targetument tfidf
        tfidf = self._init_char_ngram_tfidf(self.ngram, vocabulary)
        X_target = tfidf.fit_transform(self.target_corpus)
        ## svd
        svd = TruncatedSVD(n_components=self.svd_dim, 
                n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        svd.fit(scipy.sparse.vstack((X_obs, X_target)))
        X_obs = svd.transform(X_obs)
        X_target = svd.transform(X_target)
        #