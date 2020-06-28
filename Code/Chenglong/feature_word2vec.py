# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: word2vec based features

"""

import re
import sys
import string

import gensim
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils, pkl_utils
from utils import logging_utils, time_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper, PairwiseFeatureWrapper


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


# ------------------------ Word2Vec Features -------------------------
class Word2Vec_BaseEstimator(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, 
        aggregation_mode="", aggregation_mode_prev=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode, None, aggregation_mode_prev)
        self.model = word2vec_model
        self.model_prefix = model_prefix
        self.vector_size = word2vec_model.vector_size

    def _get_valid_word_list(self, text):
        return [w for w in text.lower().split(" ") if w in self.model]

    def _get_importance(self, text1, text2):
        len_prev_1 = len(text1.split(" "))
        len_prev_2 = len(text2.split(" "))
        len1 = len(self._get_valid_word_list(text1))
        len2 = len(self._get_valid_word_list(text2))
        imp = np_utils._try_divide(len1+len2, len_prev_1+len_prev_2)
        return imp

    def _get_n_similarity(self, text1, text2):
        lst1 = self._get_valid_word_list(text1)
        lst2 = self._get_valid_word_list(text2)
        if len(lst1) > 0 and len(lst2) > 0:
            return self.model.n_similarity(lst1, lst2)
        else:
            return config.MISSING_VALUE_NUMERIC

    def _get_n_similarity_imp(self, text1, text2):
        sim = self._get_n_similarity(text1, text2)
        imp = self._get_importance(text1, text2)
        return sim * imp

    def _get_centroid_vector(self, text):
        lst = self._get_valid_word_list(text)
        centroid = np.zeros(self.vector_size)
        for w in lst:
            centroid += self.model[w]
        if len(lst) > 0:
            centroid /= float(len(lst))
        return centroid

    def _get_centroid_vdiff(self, text1, text2):
        centroid1 = self._get_centroid_vector(text1)
        centroid2 = self._get_centroid_vector(text2)
        return dist_utils._vdiff(centroid1, centroid2)

    def _get_centroid_rmse(self, text1, text2):
        centroid1 = self._get_centroid_vector(text1)
        centroid2 = self._get_centroid_vector(text2)
        return dist_utils._rmse(centroid1, centroid2)

    def _get_centroid_rmse_imp(self, text1, text2):
        rmse = self._get_centroid_rmse(text1, text2)
        imp = self._get_importance(text1, text2)
        return rmse * imp


class Word2Vec_Centroid_Vector(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_Centroid_Vector"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_centroid_vector(obs)


class Word2Vec_Importance(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_Importance"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_importance(obs, target)


class Word2Vec_N_Similarity(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_N_Similarity"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_n_similarity(obs, target)


class Word2Vec_N_Similarity_Imp(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_N_Similarity_Imp"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_n_similarity_imp(obs, target)


class Word2Vec_Centroid_RMSE(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)
        
    def __name__(self):
        return "Word2Vec_%s_D%d_Centroid_RMSE"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_centroid_rmse(obs, target)


class Word2Vec_Centroid_RMSE_IMP(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)
        
    def __name__(self):
        return "Word2Vec_%s_D%d_Centroid_RMSE_IMP"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_centroid_rmse_imp(obs, target)


class Word2Vec_Centroid_Vdiff(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)
        
    def __name__(self):
        return "Word2Vec_%s_D%d_Centroid_Vdiff"%(self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_centroid_vdiff(obs, target)


class Word2Vec_CosineSim(Word2Vec_BaseEstimator):
    """Double aggregation features"""
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, 
        aggregation_mode="", aggregation_mode_prev=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, 
            aggregation_mode, aggregation_mode_prev)
        
    def __name__(self):
        feat_name = []
        for m1 in self.aggregation_mode_prev:
            for m in self.aggregation_mode:
                n = "Word2Vec_%s_D%d_CosineSim_%s_%s"%(
                    self.model_prefix, self