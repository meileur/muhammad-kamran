# -*- coding: utf-8 -*-
"""
Generating ensemble output from models: Igor's part.

Competition: HomeDepot Search Relevance
Author: Igor Buinyi
Team: Turing test
"""

from config_IgorKostia import *

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from time import time
import re
import os
from scipy.stats import pearsonr


df_train = pd.read_csv(DATA_DIR+'/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv(DATA_DIR+'/test.csv', encoding="ISO-8859-1")
num_train = df_train.shape[0] #number of observations
num_test = df_test.shape[0] #number of observations




dir_name=MODELSENSEMBLE_DIR


files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]


### Load files
models=[]
df_all_validation=None
#filename_from_which_id_is_read=None
for f in files:
    if f.endswith(".csv") and f.startswith("trainvalidation_"):
        model_name=f.replace("trainvalidation_","").replace(".csv","")
        submission_file=f.replace("trainvalidation_","testprediction_")
        if os.path.exists(os.path.join(dir_name, "testprediction_"+model_name+".csv")):
            df_model_validation = pd.read_csv(os.path.join(dir_name, f), index_col=False)
            df_model_submission = pd.read_csv(os.path.join(dir_name, submission_file), index_col=False)
            
            if df_all_validation is None:
                df_all_validation=df_model_validation[['id','actual']]
                df_all_submission=df_model_submission[['id']]
                filenames_from_which_ids_are_read=\
                    {'validation':f,
                     'submission':f.replace("trainvalidation_","").replace(".csv","")}
                
            if sum(df_all_validation['id']!=df_model_validation['id'])>0:
                raise ValueError("'id' column in file\n\t"+f+
                "\nis different from file \n\t"+filenames_from_which_ids_are_read['validation'])
            elif sum(df_all_validation['actual']!=df_model_validation['actual'])>0:
                raise ValueError("'actual' column in file\n\t"+f+
                "\nis different from file \n\t"+filenames_from_which_ids_are_read['validation'])
            else:
                df_all_validation[model_name]=df_model_validation['predicted']  
                
            if sum(df_all_submission['id']!=df_model_submission['id'])>0:
                raise ValueError("'id' column in file\n\t"+submission_file+
                "\nis different from file \n\t"+filenames_from_which_ids_are_read['submission'])
            else:
                df_all_submission[model_name]=df_model_submission['relevance']  
                
            models.append(model_name)
            print "loaded", model_name

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm



n=len(df_all_validation['actual'])/2

for model in models:
    print "%s\t1st split: %.5f, 2nd split: %.5f, total: %.5f" % (model, \
    mean_squared_error(df_all_validation['actual'][:n], df_all_validation[model][:n])**0.5, \
    mean_squared_error(df_all_validation['actual'][n:], df_all_validation[model][n:])**0.5,\
    mean_squared_err