# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: feature transformer

"""

from collections import Counter

from sklearn.base import BaseEstimator


#### adopted from @Ben Hamner's Python Benchmark code
## https://www.kaggle.com/benhamner/crowdflower-search-relevance/python-benchmark
def identity(x):
    return x


class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return self.transformer(X)


class ColumnSelector(BaseEstimator):
    def __init__(self, columns=-1):
        # assert (type(columns) == int) or (type(columns) == list)
        se