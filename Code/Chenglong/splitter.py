# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: splitter for Homedepot project

"""

import numpy as np
import pandas as pd
import sklearn.cross_validation
from sklearn.cross_validation import ShuffleSplit, _validate_shuffle_split
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
plt.rcParams["figure.figsize"] = [5, 5]

import config
from utils import pkl_utils


## to suppress the ValueError
class StratifiedShuffleSplit(sklearn.cross_validation.StratifiedShuffleSplit):
    def __init__(self, y, n_iter=10, test_size=0.1, train_size=None,
                 random_state=None):
        n = len(y)
        self.y = np.array(y)
        self.classes, self.y_indices = np.unique(y, return_inverse=True)
        self.random_state = random_state
        self.train_size = train_size
        self.test_size = test_size
        self.n_iter = n_iter
        self.n = n
        self.n_train, self.n_test = _validate_shuffle_split(n, test_size, train_size)


## advanced splitter
class HomedepotSplitter:
    def __init__(self, dfTrain, dfTest, n_iter=5, random_state=config.RANDOM_SEED,
                    verbose=False, plot=False, split_param=[0.5, 0.25, 0.5]):
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.plot = plot
        self.split_param = split_param

    def __str__(self):
        return "HomedepotSplitter"

    def _check_split(self, dfTrain, dfTest, col, suffix="", plot=""):
        if self.verbose:
            print("-"*50)
        num_train = dfTrain.shape[0]
        num_test = dfTest.shape[0]
        ratio_train = num_train/(num_train+num_test)
        ratio_test = num_test/(num_train+num_test)
        if self.verbose:
            print("Sample Stats: %.2f (train) | %.2f (test)" % (ratio_train, ratio_test))
        
        puid_train = set(np.unique(dfTrain[col]))
        puid_test = set(np.unique(dfTest[col]))
        puid_total = puid_train.union(puid_test)
        puid_intersect = puid_train.intersection(puid_test)
        
        ratio_train = ((len(puid_train) - len(puid_intersect)) / len(puid_total))
        ratio_intersect = len(puid_intersect) / len(puid_total)
        ratio_test = ((len(puid_test) - len(puid_intersect)) / len(puid_total))
        
        if self.verbose:
            print("%s Stats: %.2f (train) | %.2f (train & test) | %.2f (test)" % (
                col, ratio_train, ratio_intersect, ratio_test))

        if (plot == "" and self.plot) or plot:
            plt.figure()
            if suffix == "actual":
                venn2([puid_train, puid_test], ("train", "test"))
            else:
                venn2([puid_train, puid_test], ("train", "valid"))
            fig_file = "%s/%s_%s.pdf"%(config.FIG_DIR, suffix, col)
            plt.savefig(fig_file)
            plt.clf()

        ## SORT it for reproducibility !!!
        puid_train = sorted(list(puid_train))
        return puid_train

    def _get_df_idx(self, df, col, values):
        return np.where(df[col].isin(values))[0]

    def split(self):

        if self.verbose:
            print("*"*50)
            print("Original Train and Test Split")
        puid_train = self._check_split(self.dfTrain, self.dfTest, "product_uid", "actual")
        term_train = self._check_split(self.dfTrain, self.dfTest, "search_term", "actual")

        ## naive split
        if self.verbose:
            print("*"*50)
            print("Naive Split")
        rs = ShuffleSplit(n=self.dfTrain.shape[0], n_iter=1, test_size=0.69, random_state=self.random_state)
        for trainInd, validInd in rs:
            dfTrain2 = self.dfTrain.iloc[trainInd].copy()
            dfValid = self.dfTrain.iloc[validInd].copy()
            self._check_split(dfTrain2, dfValid, "product_uid", "naive")
            self._check_split(dfTrain2, dfValid, "search_term", "naive")
            
        ## split on product_uid & search_term
        if self.verbose:
            print("*"*50)
            print("Split on product_uid & search_term")
        self.splits = [0]*self.n_iter
        rs = ShuffleSplit(n=len(term_train), n_iter=self.n_iter, 
            test_size=self.split_param[0], random_state=self.random_state)
        for run, (trInd, vaInd) in enumerate(rs):
            if self.verbose:
                print("="*50)
            ntr = int(len(trInd)*self.split_param[1])
            term_train2 = [term_train[i] for i in trInd[:ntr]]
            term_common = [term_train[i] for i in trInd[ntr:]]
            term_valid = [term_train[i] for i in vaInd]
            
            trainInd = self._get_df_idx(self.dfTrain, "search_term", term_train2)
            commonInd = self._get_df_idx(self.dfTrain, "search_term", term_common)
            validInd = self._get_df_idx(self.dfTrain, "search_term", term_valid)

            sss = StratifiedShuffleSplit(self.dfTrain.iloc[commonInd]["search_term"], 
                n_iter=1, test_size=self.split_param[2], random_state=run)
            iidx, oidx = list(sss)[0]
            
            trainInd = np.hstack((trainInd, commonInd[iidx]))
            validInd = np.hstack((validInd, commonInd[oidx]))
            
            trainInd = sorted(trainInd)
            validInd = sorted(validInd)
            
            if self.verbose:
                dfTrain2 = self.dfTrain.iloc[trainInd].copy()
                dfValid = self.dfTrain.iloc[validInd].copy()
                if run == 0:
                    plot = self.plot
                else:
                    plot = False
                self._check_split(dfTrain2, dfValid, "product_uid", "proposed", plot)
                self._check_split(dfTrain2, dfValid, "search_term", "proposed", plot)
            
            self.splits[run] = trainInd, validInd
            
            if self.verbose:
                print("-"*50)
                print("Index for run: %s" % (run+1))
        