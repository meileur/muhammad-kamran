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
                    columns = ["%s_%d"%(fname, x) for x in range(dim)]
                    df = pd.DataFrame(x, columns=columns)
                    dfAll = pd.concat([dfAll, df], axis=1)
                    self.feature_names.extend(columns)
                feat_cnt += 1
                self.feature_names_basic.append(fname)
                if dim == 1:
                    self.logger.info("Combine {:>3}/{:>3} feat: {} ({}D) (corr = {})".format(
                        feat_cnt, len(self.feature_dict.keys()), fname, dim, corr))
                else:
                    self.logger.info("Combine {:>3}/{:>3} feat: {} ({}D)".format(
                        feat_cnt, len(self.feature_dict.keys()), fname, dim))
        dfAll.fillna(config.MISSING_VALUE_NUMERIC, inplace=True)
        ## basic
        dfTrain = dfAll.iloc[:TRAIN_SIZE].copy()
        self.y_train = dfTrain["relevance"].values.astype(float)
        dfTrain.drop(["id","relevance"], axis=1, inplace=True)
        self.X_train = dfTrain.values.astype(float)

        dfTest = dfAll.iloc[TRAIN_SIZE:].copy()
        self.id_test = dfTest["id"].values.astype(int)
        dfTest.drop(["id","relevance"], axis=1, inplace=True)
        self.X_test = dfTest.values.astype(float)

        ## all
        first = True
        feat_cv_cnt = 0
        dfAll_cv_all = dfAll_raw.copy()
        feature_dir = "%s/All" % (config.FEAT_DIR)
        for file_name in sorted(os.listdir(feature_dir)):
            if self.feature_suffix in file_name:
                fname = file_name.split(".")[0]
                if fname not in self.feature_dict:
                    continue
                if first:
                    self.logger.info("Run for all...")
                    first = False
                x = self.load_feature(feature_dir, fname)
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
                    dfAll_cv_all[fname] = x
                    self.feature_names.append(fname)
                else:
                    columns = ["%s_%d"%(fname, x) for x in range(dim)]
                    df = pd.DataFrame(x, columns=columns)
                    dfAll_cv_all = pd.concat([dfAll_cv_all, df], axis=1)
                    self.feature_names.extend(columns)
                feat_cv_cnt += 1
                self.feature_names_cv.append(fname)
                if dim == 1:
                    self.logger.info("Combine {:>3}/{:>3} feat: {} ({}D) (corr = {})".format(
                        feat_cnt+feat_cv_cnt, len(self.feature_dict.keys()), fname, dim, corr))
                else:
                    self.logger.info("Combine {:>3}/{:>3} feat: {} ({}D)".format(
                        feat_cnt+feat_cv_cnt, len(self.feature_dict.keys()), fname, dim))
        if feat_cv_cnt > 0:
            dfAll_cv_all.fillna(config.MISSING_VALUE_NUMERIC, inplace=True)
            X_tmp = dfAll_cv_all.drop(["id","relevance"], axis=1).values.astype(float)
            self.X_train_cv_all = X_tmp[:TRAIN_SIZE]
            self.X_test = np.hstack((self.X_test, X_tmp[TRAIN_SIZE:]))
        else:
            self.X_train_cv_all = None
        feat_cnt += feat_cv_cnt

        ## for cv features
        first = True
        for run in range(1,self.n_iter+1):
            feat_cv_cnt = 0
            dfAll_cv = dfAll_raw.copy()
            feature_dir = "%s/Run%d" % (config.FEAT_DIR, run)
            for file_name in sorted(os.listdir(feature_dir)):
                if self.feature_suffix in file_name:
                    fname = file_name.split(".")[0]
                    if (fname not in self.feature_dict) or (fname not in self.feature_names_cv):
                        continue
                    if first:
                        self.logger.info("Run for cv...")
                        first = False
                    if feat_cv_cnt == 0:
                        self.logger.info("Run %d"%run)
                    x = self.load_feature(feature_dir, fname)
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
                        dfAll_cv[fname] = x
                    else:
                        columns = ["%s_%d"%(fname, x) for x in range(dim)]
                        df = pd.DataFrame(x, columns=columns)
                        dfAll_cv = pd.concat([dfAll_cv, df], axis=1)
                    feat_cv_cnt += 1
                    self.logger.info("Combine {:>3}/{:>3} feat: {} ({}D)".format(
                        feat_cnt+feat_cv_cnt, len(self.feature_dict.keys()), fname, dim))
            if feat_cv_cnt > 0:
                dfAll_cv.fillna(config.MISSING_VALUE_NUMERIC, inplace=True)
                dfTrain_cv = dfAll_cv.iloc[:TRAIN_SIZE].copy()
                X_tmp = dfTrain_cv.drop(["id","relevance"], axis=1).values.astype(float)
                if run == 1:
                    self.X_train_cv = np.zeros((X_tmp.shape[0], X_tmp.shape[1], self.n_iter), dtype=float)
                self.X_train_cv[:,:,run-1] = X_tmp
        if feat_cv_cnt == 0:
            self.X_train_cv = None
            self.basic_only = 1

        # report final results
        if self.basic_only:
            self.logger.info("Overall Shape: %d x %d"%(len(self.y_train), self.X_train.shape[1]))
        else:
            self.logger.info("Overall Shape: %d x %d"%(
                len(self.y_train), self.X_train.shape[1]+self.X_train_cv_all.shape[1])) 
        self.logger.info("Done combinning.")

        return self

    def save(self):
        data_dict = {
            "X_train_basic": self.X_train,
            "y_train": self.y_train,
            "X_train_cv": self.X_train_cv,
            "X_train_cv_all": self.X_train_cv_all,
            "X_test": self.X_test,                    
            "id_test": self.id_test,
            "splitter": self.splitter,
            "n_iter": self.n_iter,
            "basic_only": self.basic_only,
            "feature_names": self.feature_names
        }
        fname = os.path.join(config.FEAT_DIR+"/Combine", self.feature_name+config.FEAT_FILE_SUFFIX)
        pkl_utils._save(fname, data_dict)
        self.logger.info("Save to %s" % fname)


class StackingCombiner:
    def __init__(self, feature_list, feature_name, feature_suffix=".csv",
                feature_level=2, meta_feature_dict={}, corr_threshold=0):
        self.feature_name = feature_name
        self.feature_list = feature_list
        self.feature_suffix = feature_suffix
        self.feature_level = feature_level
        # for meta features
        self.meta_feature_dict = meta_feature_dict
        self.corr_threshold = corr_threshold
        self.feature_names_basic = []
        self.feature_names_cv = []
        self.feature_names = []
        self.has_basic = 1 if self.meta_feature_dict else 0
        logname = "feature_combiner_%s_%s.log"%(feature_name, time_utils._timestamp())
        self.logger = logging_utils._get_logger(config.LOG_DIR, logname)
        if self.feature_level == 2:
            self.splitter = splitter_level2
        elif self.feature_level == 3:
            self.splitter = splitter_level3
        self.n_iter = n_iter
        self.splitter_prev = [0]*self.n_iter

    def load_feature(self, feature_dir, feature_name, columns="prediction", columns_pattern=""):
        fname = os.path.join(feature_dir, feature_name+self.feature_suffix)
        df = pd.read_csv(fname)
        if columns is None or columns == "" or len(columns) == 0:
            columns = [col for col in df.columns if columns_pattern in col]
        return df[columns].values

    def combine(self):
        # combine meta features
        if self.meta_feature_dict:
            cb = Combiner(feature_dict=self.meta_feature_dict, 
                        feature_name=self.feature_name, 
                        feature_suffix=".pkl