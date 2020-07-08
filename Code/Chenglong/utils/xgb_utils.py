
# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for XGBoost models

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import xgboost as xgb


class XGBRegressor:
    def __init__(self, booster='gbtree', base_score=0., colsample_bylevel=1., 
                colsample_bytree=1., gamma=0., learning_rate=0.1, max_delta_step=0.,
                max_depth=6, min_child_weight=1., missing=None, n_estimators=100, 
                nthread=1, objective='reg:linear', reg_alpha=1., reg_lambda=0., 
                reg_lambda_bias=0., seed=0, silent=True, subsample=1.):
        self.param = {
            "objective": objective,
            "booster": booster,
            "eta": learning_rate,
            "max_depth": max_depth,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "alpha": reg_alpha,
            "lambda": reg_lambda,
            "lambda_bias": reg_lambda_bias,
            "seed": seed,
            "silent": 1 if silent else 0,
            "nthread": nthread,
            "max_delta_step": max_delta_step,
        }
        self.missing = missing if missing is not None else np.nan
        self.n_estimators = n_estimators
        self.base_score = base_score

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ("%s(booster=\'%s\', base_score=%f, colsample_bylevel=%f, \n"
                    "colsample_bytree=%f, gamma=%f, learning_rate=%f, max_delta_step=%f, \n"
                    "max_depth=%d, min_child_weight=%f, missing=\'%s\', n_estimators=%d, \n"
                    "nthread=%d, objective=\'%s\', reg_alpha=%f, reg_lambda=%f, \n"
                    "reg_lambda_bias=%f, seed=%d, silent=%d, subsample=%f)" % (
                    self.__class__.__name__,
                    self.param["booster"],
                    self.base_score,
                    self.param["colsample_bylevel"],
                    self.param["colsample_bytree"],
                    self.param["gamma"],
                    self.param["eta"],
                    self.param["max_delta_step"],
                    self.param["max_depth"],
                    self.param["min_child_weight"],
                    str(self.missing),
                    self.n_estimators,
                    self.param["nthread"],