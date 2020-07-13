# -*- coding: utf-8 -*-
"""
Generate features.

Competition: HomeDepot Search Relevance
Author: Igor Buinyi
Team: Turing test
"""

from config_IgorKostia import *

import numpy as np
import pandas as pd
from time import time
import re
import csv
import os
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
stoplist.append('till')
stoplist_wo_can=stoplist[:]
stoplist_wo_can.remove('can')

from homedepot_functions import *


t0 = time()
t1 = time()

#################################################################
### STEP 0: Load the results of text preprocessing
#################################################################

df_pro_desc = pd.read_csv(PROCESSINGTEXT_DIR+'/df_product_descriptions_processed.csv')
df_attr_bullets = pd.read_csv(PROCESSINGTEXT_DIR+'/df_attribute_bullets_processed.csv')
df_all = pd.read_csv(PROCESSINGTEXT_DIR+'/df_train_and_test_processed.csv')
print 'loading time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()



df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_attr_bullets, how='left', on='product_uid')
print 'merging time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()



for var in df_all.keys():
    df_all[var]=df_all[var].fillna("") 




### the function returns text after a specific word
### for example, extract_after_word('faucets for kitchen') 
### will return 'kitchen'
def extract_after_word(s,word):
    output=""
    if word in s:
        srch= re.search(r'(?<=\b'+word+'\ )[a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]+',s)
        if srch!=None:
            output=srch.group(0)
    return output

### get 'for' and 'with' parts from query and 'without' from product title
df_all['search_term_for']=df_all['search_term_parsed'].map(lambda x: extract_after_word(x,'for'))
df_all['search_term_for_stemmed']=df_all['search_term_for'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))

df_all['search_term_with']=df_all['search_term_parsed'].map(lambda x: extract_after_word(x,'with'))
df_all['search_term_with_stemmed']=df_all['search_term_with'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))

df_all['product_title_parsed_without']=df_all['product_title_parsed'].map(lambda x: extract_after_word(x,'without'))
df_all['product_title_without_stemmed']=df_all['product_title_parsed_without'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))



### save the list of string variables which are not features
string_variables_list=list(df_all.keys())
print str(len(string_variables_list) ) + " total variables..."
string_variables_list.remove('id')
string_variables_list.remove('product_uid')
string_variables_list.remove('relevance')
string_variables_list.remove('is_query_misspelled')
print "including "+ str(len(string_variables_list) ) + " text variables to drop later"



t0 = time()
#################################################################
### STEP 1: Dummy for no attributes and empty attribute bullets
#################################################################
df_attr_bullets['has_attributes_dummy']=1
df_all = pd.merge(df_all, df_attr_bullets[['product_uid','has_attributes_dummy']], how='left', on='product_uid')
df_all['has_attributes_dummy']= df_all['has_attributes_dummy'].fillna(0)


df_all['no_bullets_dummy'] = df_all['attribute_bullets'].map(lambda x:int(len(x)==0))
from google_dict import *
df_all['is_replaced_using_google_dict']=df_all['search_term'].map(lambda x: 1 if x in google_dict.keys() else 0)



df_attr_bullets=df_attr_bullets.drop(list(df_attr_bullets.keys()),axis=1)
df_pro_desc=df_pro_desc.drop(list(df_pro_desc.keys()),axis=1)


#################################################################
### STEP 2: Basic text features
#################################################################

### Calculated here are basic text features:
### * number of words/digits
### * average length of word
### * ratio of vowels in query
### * average length of word
### * number/length of brands/materials
### * percentage of digits


def sentence_statistics(s):
    s= re.sub('[^a-zA-Z0-9\ \%\$\-]', '', s)
    word_list=s.split()
    meaningful_word_list=[word for word in s.split() if len(re.findall(r'\d+', word))==0 and len(wn.synsets(word))>0]
    vowels=sum([len(re.sub('[^aeiou]', '', word)) for word in word_list])
    letters = sum([len(word) for word in word_list])
    
    return len(word_list), len(meaningful_word_list), 1.0*sum([len(word) for word in word_list])/len(word_list), 1.0*vowels/letters

df_all['query_sentence_stats_tuple'] = df_all['search_term_parsed'].map(lambda x:  sentence_statistics(x) )
df_all['len_of_meaningful_words_in_query'] = df_all['query_sentence_stats_tuple'].map(lambda x:  x[1] )
df_all['ratio_of_meaningful_words_in_query'] = df_all['query_sentence_stats_tuple'].map(lambda x:  1.0*x[1]/x[0] )
df_all['avg_wordlength_in_query'] = df_all['query_sentence_stats_tuple'].map(lambda x:  x[2] )
df_all['ratio_vowels_in_query'] = df_all['query_sentence_stats_tuple'].map(lambda x:  x[3] )
df_all=df_all.drop(['query_sentence_stats_tuple'],axis=1)


df_all['initial_len_of_query'] = df_all['search_term_stemmed'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_query_woBM'] = df_all['search_term_stemmed_woBM'].map(lambda x:len(x.split())).astype(np.int64)

df_all['len_of_query_for'] = df_all['search_term_for_stemmed'].map(lambda x:len(words_wo_digits(x, minLength=1).split())).astype(np.int64)
df_all['len_of_query_with'] = df_all['search_term_with_stemmed'].map(lambda x:len(words_wo_digits(x, minLength=1).split())).astype(np.int64)
df_all['len_of_prtitle_without'] = df_all['product_title_without_stemmed'].map(lambda x:len(words_wo_digits(x, minLength=1).split())).astype(np.int64)

df_all['len_of_query_string_only_woBM'] = df_all['search_term_stemmed_woBM'].map(lambda x:len(words_wo_digits(x, minLength=1).split())).astype(np.int64)

df_all['len_of_query_w_dash_woBM'] = df_all['search_term_stemmed_woBM'].map(lambda x:len(words_w_dash(x).split())).astype(np.int64)
df_all['len_of_title_w_dash_woBM'] = df_all['product_title_stemmed_woBM'].map(lambda x:len(words_w_dash(x).split())).astype(np.int64)

df_all['len_of_product_title_woBM'] = df_all['product_title_stemmed_woBM'].map(lambda x:len(x.split())).astype(np.int64)

df_all['len_of_product_description_woBM'] = df_all['product_description_stemmed_woBM'].map(lambda x:len(x.split())).astype(np.int64)

df_all['len_of_attribute_bullets_woBM'] = df_all['attribute_bullets_stemmed_woBM'].map(lambda x:len(x.split())).astype(np.int64)

df_all['len_of_brands_in_query'] = df_all['brands_in_search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_brands_in_query'] = df_all['brands_in_search_term'].map(lambda x:len(x.split(";"))).astype(np.int64)
df_all['len_of_materials_in_query'] = df_all['materials_in_search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_materials_in_query'] = df_all['materials_in_search_term'].map(lambda x:len(x.split(";"))).astype(np.int64)

df_all['len_of_brands_in_product_title'] = df_all['brands_in_product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_brands_in_product_title'] = df_all['brands_in_product_title'].map(lambda x:len(x.split(";")