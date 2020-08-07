# -*- coding: utf-8 -*-
"""
Code for calculating gramms and terms and some dist and TFIDF features from them.
Competition: HomeDepot Search Relevance
Author: Kostia Omelianchuk
Team: Turing test
"""

from config_IgorKostia import *

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
import nltk
from time import time
import re
import os
import math as m
import gc
import sys

from homedepot_functions import str_stemmer


###data loading
df_all=pd.read_csv(PROCESSINGTEXT_DIR+"/df_train_and_test_processed.csv", encoding="ISO-8859-1")
df_all1=pd.read_csv(PROCESSINGTEXT_DIR+"/df_product_descriptions_processed.csv", encoding="ISO-8859-1")
df_all2 = pd.merge(df_all, df_all1, how="left", on="product_uid")
df_all = df_all2
df_all1=pd.read_csv(PROCESSINGTEXT_DIR+"/df_attribute_bullets_processed.csv", encoding="ISO-8859-1")
df_all2 = pd.merge(df_all, df_all1, how="left", on="product_uid")
df_all = df_all2

def replace_nan(s):
        if pd.isnull(s)==True:
                s=""
        return s
#code for attributes creation
df_attr = pd.read_csv(DATA_DIR+'/attributes.csv', encoding="ISO-8859-1")
def replace_nan_float(s):
        if np.isnan(s)==True:
                s=0
        return s
df_attr['product_uid'] = df_attr['product_uid'].map(lambda x:replace_nan_float(x))
df_attr['name'] = df_attr['name'].map(lambda x:replace_nan(x))
df_attr['value'] = df_attr['value'].map(lambda x:replace_nan(x))

pid = list(set(list(df_attr["product_uid"])))
#name= list(set(list(df_attr["name"])))

df_attr["all"]=df_attr["name"]+" "+df_attr['value']
df_attr['all'] = df_attr['all'].map(lambda x:replace_nan(x))

at=list()
for i in range(len(pid)):
    at.append(' '.join(list(df_attr["all"][df_attr["product_uid"]==pid[i]])))

df_atrr = pd.DataFrame({'product_uid' : pd.Series(pid[1:]), 'value' : pd.Series(at[1:])})

#use Igor stemmer for process attributes from 'homedepot_fuctions.py'
df_atrr['attribute_stemmed']=df_atrr['value'].map(lambda x:str_stemmer(x))
df_atrr.to_csv(PROCESSINGTEXT_DIR+"/df_attributes_kostia.csv",  index=False, encoding="utf-8") 




#df_attr = pd.read_csv(DATA_DIR+'/df_attributes_kostia.csv', encoding="utf-8")
df_all = pd.merge(df_all, df_atrr, how='left', on='product_uid')


p = df_all.keys()
for i in range(len(p)):
    print p[i]


#replace nan
df_all['search_term'] = df_all['search_term'].map(lambda x:replace_nan(x))
df_all['produc