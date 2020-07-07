# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: definitions for
        - learner & ensemble learner
        - feature & stacking feature
        - task & stacking task
        - task optimizer

"""

import os
import sys
import time
from optparse import OptionParser

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

import config
from utils import dist_utils, logging_utils, pkl_utils, time_utils
from utils.xgb_utils import XGBRegressor, HomedepotXGBClassifier as XGBClassifier
from utils.rgf_utils import RGFRegressor
from utils.skl_utils import SVR, LinearSVR, KNNRegressor, AdaBoostRegressor, RandomRidge
try:
    from utils.keras_utils import KerasDNNRegressor
except:
    pass
from model_param_space import ModelParamSpace


class Learner:
    def __init__(self, learner_name, param_dict):
        self.learner_name = learner_name
        self.param_dict = param_dict
        self.learner = self._get_learner()

    def __str__(self):
        return self.learner_name

    def _get_learner(self):
        # xgboost
        if self.learner_name in ["reg_xgb_linear", "reg_xgb_tree", "reg_xgb_tree_best_single_model"]:
            return XGBRegressor(**self.param_dict)
        if self.learner_name in ["clf_xgb_linear", "clf_xgb_tree"]:
            return XGBClassifier(**self.param_dict)
        # sklearn
        if self.learner_name == "reg_skl_lasso":
            return Lasso(**self.param_dict)
        if self.learner_name == "reg_skl_ridge":
            return Ridge(**self.param_dict)
        if self.learner_name == "reg_skl_random_ridge":
            return RandomRidge(**self.param_dict)
        if self.learner_name == "reg_skl_bayesian_ridge":
            return BayesianRidge(**self.param_dict)
        if self.learner_name == "reg_skl_svr":
            return SVR(**self.param_dict)
        if self.learner_name == "reg_skl_lsvr":
            return LinearSVR(**self.param_dict)
        if self.learner_name == "reg_skl_knn":
            return KNNRegressor(**self.param_dict)
        if self.learner_name == "reg_skl_etr":
            return ExtraTreesRegressor(**self.param_dict)
        if self.learner_name == "reg_skl_rf":
            return RandomForestRegressor(**self.param_dict)
        if self.learner_name == "reg_skl_gbm":
            return GradientBoostingRegressor(**self.param_dict)
        if self.learner_name == "reg_skl_adaboost":
            return AdaBoostRegressor(**self.param_dict)
        # keras
        if self.learner_name == "reg_keras_dnn":
            try:
                return KerasDNNRegressor(**self.param_dict)
            except:
                return None
        # rgf
        if self.learner_name == "reg_rgf":
            return RGFRegressor(**self.param_dict)
        # ensemble
        if self.learner_name == "reg_ensemble":
            return EnsembleLearner(**self.param_dict)
            
        return None

    def fit(self, X, y, feature_names=None):
        if feature_names is not None:
            self.learner.fit(X, y, feature_names)
        else:
            self.learner.fit(X, y)
        return self

    def predict(self, X, feature_names=None):
        if feature_names is not None:
            y_pred = self.learner.predict(X, feature_names)
        else:
            y_pred = self.learner.predict(X)
        # relevance is in [1,3]
        y_pred = np.clip(y_pred, 1., 3.)
        return y_pred

    def plot_importance(self):
        ax = self.learner.plot_importance()
        return ax


class EnsembleLearner:
    def __init__(self, learner_dict):
        self.learner_dict = learner_dict

    def __str__(self):
        return "EnsembleLearner"

    def fit(self, X, y):
        for learner_name in self.learner_dict.keys():
            p = self.learner_dict[learner_name]["param"]
            l = Learner(learner_name, p)._get_learner()
            if l is not None:
                self.learner_dict[learner_name]["learner"] = l.fit(X, y)
            else:
                self.learner_dict[learner_name]["learner"] = None
        return self

    def predict(self, X):
        y_pred = np.zeros((X.shape[0]), dtype=float)
        w_sum = 0.
        for learner_name in self.learner_dict.keys():
            l = self.learner_dict[learner_name]["learner"]
            if l is not None:
                w = self.learner_dict[learner_name]["weight"]
                y_pred += w * l.predict(X)
                w_sum += w
        y_pred /= w_sum
        return y_pred


class Feature:
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.data_dict = self._load_data_dict()
        self.splitter = self.data_dict["splitter"]
        self.n_iter = self.data_dict["n_iter"]

    def __str__(self):
        return self.feature_name

    def _load_data_dict(self):
        fname = os.path.join(config.FEAT_DIR+"/Combine", self.feature_name+config.FEAT_FILE_SUFFIX)
        data_dict = pkl_utils._load(fname)
        return data_dict

    ## for CV
    def _get_train_valid_data(self, i):
        # feature
        X_basic_train = self.data_dict["X_train_basic"][self.splitter[i][0], :]
        X_basic_valid = self.data_dict["X_train_basic"][self.splitter[i][1], :]
        if self.data_dict["basic_only"]:
            X_train, X_valid = X_basic_train, X_basic_valid
        else:
            X_train_cv = self.data_dict["X_train_cv"][self.splitter[i][0], :, i]
            X_valid_cv = self.data_dict["X_train_cv"][self.splitter[i][1], :, i]
            X_train = np.hstack((X_basic_train, X_train_cv))
            X_valid = np.hstack((X_basic_valid, X_valid_cv))
        # label
        y_train = self.data_dict["y_train"][self.splitter[i][0]]
        y_valid = self.data_dict["y_train"][self.splitter[i][1]]

        return X_train, y_train, X_valid, y_valid

 