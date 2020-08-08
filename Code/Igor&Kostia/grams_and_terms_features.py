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
df_all['product_title'] = df_all['product_title'].map(lambda x:replace_nan(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:replace_nan(x))
df_all['attribute_bullets'] = df_all['attribute_bullets'].map(lambda x:replace_nan(x))
df_all['value'] = df_all['value'].map(lambda x:replace_nan(x))
   
df_all['search_term_stemmed'] = df_all['search_term_stemmed'].map(lambda x:replace_nan(x))
df_all['product_title_stemmed'] = df_all['product_title_stemmed'].map(lambda x:replace_nan(x))
df_all['product_description_stemmed'] = df_all['product_description_stemmed'].map(lambda x:replace_nan(x))
df_all['attribute_bullets_stemmed'] = df_all['attribute_bullets_stemmed'].map(lambda x:replace_nan(x))
df_all['attribute_stemmed'] = df_all['attribute_stemmed'].map(lambda x:replace_nan(x))


#create one big string
df_all['product_info'] = df_all['search_term']+"\t"+df_all['search_term_stemmed']+ "\t"  \
             +df_all['product_title']+"\t"+df_all['product_title_stemmed'] + "\t"\
             +df_all['product_description'] + "\t"+df_all['product_description_stemmed'] + "\t"\
             +df_all['attribute_bullets']+ "\t"+df_all['attribute_bullets_stemmed']  + "\t"\
             +df_all['value'] + "\t"+df_all['attribute_stemmed']

df_all['product_info'] = df_all['product_info'].map(lambda x:replace_nan(x))


#gram and terms functions
def getUnigram(str1):
    """
        Input: a list of words, e.g., ['I', 'am', 'Denny']
        Output: a list of unigram
    """
    words = str1.split()
    assert type(words) == list
    return words
    


def getBigram(str1, join_string, skip=0):
	"""
	   Input: a list of words, e.g., ['I', 'am', 'Denny']
	   Output: a list of bigram, e.g., ['I_am', 'am_Denny']
	   I use _ as join_string for this example.
	"""
	words = str1.split()
	assert type(words) == list
	L = len(words)
	if L > 1:
		lst = []
		for i in range(L-1):
			for k in range(1,skip+2):
				if i+k < L:
					lst.append( join_string.join([words[i], words[i+k]]) )
	else:
		# set it as unigram
		lst = getUnigram(str1)
	return lst


   
def getTrigram(str1, join_string, skip=0):
	"""
	   Input: a list of words, e.g., ['I', 'am', 'Denny']
	   Output: a list of trigram, e.g., ['I_am_Denny']
	   I use _ as join_string for this example.
	"""
	words = str1.split()
	assert type(words) == list
	L = len(words)
	if L > 2:
		lst = []
		for i in range(L-2):
			for k1 in range(1,skip+2):
				for k2 in range(1,skip+2):
					if i+k1 < L and i+k1+k2 < L:
						lst.append( join_string.join([words[i], words[i+k1], words[i+k1+k2]]) )
	else:
		# set it as bigram
		lst = getBigram(str1, join_string, skip)
	return lst
    
def getFourgram(str1, join_string):
    """
        Input: a list of words, e.g., ['I', 'am', 'Denny', 'boy']
        Output: a list of trigram, e.g., ['I_am_Denny_boy']
        I use _ as join_string for this example.
    """
    words = str1.split()
    assert type(words) == list
    L = len(words)
    if L > 3:
        lst = []
        for i in xrange(L-3):
            lst.append( join_string.join([words[i], words[i+1], words[i+2], words[i+3]]) )
    else:
        # set it as bigram
        lst = getTrigram(str1, join_string)
    return lst

def getBiterm(str1, join_string):
    """
        Input: a list of words, e.g., ['I', 'am', 'Denny', 'boy']
        Output: a list of biterm, e.g., ['I_am', 'I_Denny', 'I_boy', 'am_Denny', 'am_boy', 'Denny_boy']
        I use _ as join_string for this example.
    """
    words = str1.split()
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for j in range(i+1,L):
                lst.append( join_string.join([words[i], words[j]]) )
    else:
        # set it as unigram
        lst = getUnigram(str1)
    return lst
    
def getTriterm(str1, join_string):
    """
        Input: a list of words, e.g., ['I', 'am', 'Denny']
        Output: a list of triterm, e.g., ['I_am_Denny', 'I_Denny_am', 'am_I_Denny',
        'am_Denny_I', 'Denny_I_am', 'Denny_am_I']
        I use _ as join_string for this example.
    """
    words = str1.split()
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in xrange(L-2):
            for j in xrange(i+1,L-1):
                for k in xrange(j+1,L):
                    lst.append( join_string.join([words[i], words[j], words[k]]) )
    else:
        # set it as biterm
        lst = getBiterm(str1, join_string)
    return lst

gc.collect()

#make gramms and terms (need a lot of memory)
t0 = time()
df_all['st_unigram'] = df_all['product_info'].map(lambda x:getUnigram(x.split('\t')[1]))
df_all['pt_unigram'] = df_all['product_info'].map(lambda x:getUnigram(x.split('\t')[3]))
df_all['pd_unigram'] = df_all['product_info'].map(lambda x:getUnigram(x.split('\t')[5]))
df_all['ab_unigram'] = df_all['product_info'].map(lambda x:getUnigram(x.split('\t')[7]))
df_all['at_unigram'] = df_all['product_info'].map(lambda x:getUnigram(x.split('\t')[9]))
print 'unigram time:',round(time()-t0,3) ,'s\n'

st_names=["st_unigram", "pt_unigram","pd_unigram","ab_unigram","at_unigram"]
b=df_all[st_names]
b["id"]=df_all["id"]
b.to_csv(PROCESSINGTEXT_DIR+"/df_unigram.csv", index=False, encoding='utf-8') 
#df_all=df_all.drop(st_names,axis=1)
print 1
gc.collect()
t0 = time()
df_all['st_bigram'] = df_all['product_info'].map(lambda x:getBigram(x.split('\t')[1],"_"))
df_all['pt_bigram'] = df_all['product_info'].map(lambda x:getBigram(x.split('\t')[3],"_"))
df_all['pd_bigram'] = df_all['product_info'].map(lambda x:getBigram(x.split('\t')[5],"_"))
df_all['ab_bigram'] = df_all['product_info'].map(lambda x:getBigram(x.split('\t')[7],"_"))
df_all['at_bigram'] = df_all['product_info'].map(lambda x:getBigram(x.split('\t')[9],"_"))
print 'bigram time:',round(time()-t0,3) ,'s\n'

st_names=["st_bigram", "pt_bigram","pd_bigram","ab_bigram","at_bigram"]
b=df_all[st_names]
b["id"]=df_all["id"]
b.to_csv(PROCESSINGTEXT_DIR+"/df_bigram.csv", index=False, encoding='utf-8') 
#df_all=df_all.drop(st_names,axis=1)
print 2
gc.collect()
t0 = time()
df_all['st_trigram'] = df_all['product_info'].map(lambda x:getTrigram(x.split('\t')[1],"_"))
df_all['pt_trigram'] = df_all['product_info'].map(lambda x:getTrigram(x.split('\t')[3],"_"))
df_all['pd_trigram'] = df_all['product_info'].map(lambda x:getTrigram(x.split('\t')[5],"_"))
df_all['ab_trigram'] = df_all['product_info'].map(lambda x:getTrigram(x.split('\t')[7],"_"))
df_all['at_trigram'] = df_all['product_info'].map(lambda x:getTrigram(x.split('\t')[9],"_"))
print 'trigram time:',round(time()-t0,3) ,'s\n'

st_names=["st_trigram", "pt_trigram","pd_trigram","ab_trigram","at_trigram"]
b=df_all[st_names]
b["id"]=df_all["id"]
b.to_csv(PROCESSINGTEXT_DIR+"/df_trigram.csv", index=False, encoding='utf-8') 
#df_all=df_all.drop(st_names,axis=1)
print 3
gc.collect()
t0 = time()
df_all['st_fourgram'] = df_all['product_info'].map(lambda x:getFourgram(x.split('\t')[1],"_"))
df_all['pt_fourgram'] = df_all['product_info'].map(lambda x:getFourgram(x.split('\t')[3],"_"))
df_all['pd_fourgram'] = df_all['product_info'].map(lambda x:getFourgram(x.split('\t')[5],"_"))
df_all['ab_fourgram'] = df_all['product_info'].map(lambda x:getFourgram(x.split('\t')[7],"_"))
df_all['at_fourgram'] = df_all['product_info'].map(lambda x:getFourgram(x.split('\t')[9],"_"))
print 'fourgram time:',round(time()-t0,3) ,'s\n'

st_names=["st_fourgram", "pt_fourgram","pd_fourgram","ab_fourgram","at_fourgram"]
b=df_all[st_names]
b["id"]=df_all["id"]
b.to_csv(PROCESSINGTEXT_DIR+"/df_fourgram.csv", index=False, encoding='utf-8') 
#df_all=df_all.drop(st_names,axis=1)
print 4
gc.collect()
t0 = time()
df_all['st_biterm'] = df_all['product_info'].map(lambda x:getBiterm(x.split('\t')[1],"_"))
df_all['pt_biterm'] = df_all['product_info'].map(lambda x:getBiterm(x.split('\t')[3],"_"))

#df_all['pd_biterm'] = df_all['product_info'].map(lambda x:getBiterm(x.split('\t')[5],"_"))
#df_all['ab_biterm'] = df_all['product_info'].map(lambda x:getBiterm(x.split('\t')[7],"_"))
#df_all['at_biterm'] = df_all['product_info'].map(lambda x:getBiterm(x.split('\t')[9],"_"))
print 'biterm time:',round(time()-t0,3) ,'s\n'

#st_names=["st_biterm", "pt_biterm","pd_biterm","ab_biterm","at_biterm"]
#b=df_all[st_names]
#b["id"]=df_all["id"]
#b.to_csv(DATA_DIR+"/df_biterm.csv", index=False) 
#df_all=df_all.drop(st_names,axis=1)
#print 5

gc.collect()
t0 = time()
df_all['st_triterm'] = df_all['product_info'].map(lambda x:getTriterm(x.split('\t')[1],"_"))
df_all['pt_triterm'] = df_all['product_info'].map(lambda x:getTriterm(x.split('\t')[3],"_"))
#df_all['pd_triterm'] = df_all['product_info'].map(lambda x:getTriterm(x.split('\t')[5],"_"))
#df_all['ab_triterm'] = df_all['product_info'].map(lambda x:getTriterm(x.split('\t')[7],"_"))
#df_all[