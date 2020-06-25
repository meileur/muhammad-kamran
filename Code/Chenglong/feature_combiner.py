# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: feature combiner

"""

import os
import sys
import imp
from optparse import OptionParser

import scipy
import pandas as pd
import numpy as np

import config
from config import TRAIN_SIZE
from utils import logging_utils, time_utils, pkl_utils, np_utils


splitter_level1 = pkl_utils._load("%s/splits_level1.pkl"%config.SPLIT_DIR)
splitter_level2 = pkl_utils._load("%s/splits_level2.pkl"%config.SPLIT_DIR)
splitter_level3 = pkl_utils._load("%s/splits_level3.pkl"%config.SPLIT_DIR)
assert len(splitter_level1) == len(splitter_level2)
assert len(splitter_level1) == len(splitter_level3)
n_iter = len(splitter_level1)


class Combiner:
    def __init__(self, feature_dict, feature_name, feature_suffix=".pkl", corr_threshold=0):
        self.feature_name = feature_name
        self.feature_dict = feature_dict
        self.feature_suffix = feature_suffix
        self.corr_threshold = corr_threshold
        self.feature_names_basic = []
        self.feature_names_cv = []
        self.feature_names = []
        self.basic_only = 0
        logname = "feature_combiner_%s_%s.log"%(feature_name, time_utils._timestamp())
        self.logger = logging_utils._get_logger(config.LOG_DIR, logname)
        self.splitter = splitter_level1
        self.n_iter = n_iter

    def load_feature(self, feature_dir, feature_name):
        fname = os.path.join(feature_dir, feature_name+self.feature_suffix)
        return pkl_utils._load(fname)

    def combine(self):

        dfAll = pkl_utils._load(config.INFO_DATA)
        dfAll_raw = dfAll.copy()
        y_train = dfAll["relevance"].values[:TRAIN_SIZE]

        ## for basic features
        feat_cnt = 0
        self.logger.info("Run for basic...")
        for file_name in sorted(os.listdir(config.FEAT_DIR)):
            if self.feature_suffix in file_name:
                fname = file_name.split(".")[0]
                if fname not in self.feature_dict:
                    continue
                x = self.load_feature(config.FEAT_DIR, fname)
                x = np.nan_to_num(x)
                if np.isnan(x).any():
                    self.logger.info("%s nan"%fname)
                    continue
                # apply feature transform
                mandatory = self.feature_dict[fname][0]
                transformer = self.feature_dict[fname][1]
                x = transformer.fit_transform(x)
                dim = np_utils._dim(x)
                if dim == 1:
                    corr = np_utils._corr(x[:TRAIN_SIZE], y_train)
                    if not mandatory and abs(corr) < self.corr_threshold:
                        self.logger.info("Drop: {} ({}D) (abs corr = {}, < threshold = {})".format(
                            fname, dim, abs(corr), self.corr_threshold))
                        continue
                    dfAll[fname] = x
                    self.feature_names.append(fname)
                else:
                    columns = ["%s_%d"%(fnam