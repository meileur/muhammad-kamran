
# -*- coding: utf-8 -*-
"""
Initial text preprocessing.
Although text processing can be technically done within feature generation functions, 
we found it to be very efficient to make all preprocessing first and only then move to 
feature generation. It is because the same processed text is used as an input to
generate several different features. 

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
stoplist.append('till')  # add 'till' to stoplist

# 'can' also might mean 'a container' like in 'trash can' 
# so we create a separate stop list without 'can' to be used for query and product title
stoplist_wo_can=stoplist[:]
stoplist_wo_can.remove('can')


from homedepot_functions import *
from google_dict import *

t0 = time()
t1 = time()


############################################
####### PREPROCESSING ######################
############################################

### load train and test ###################
df_train = pd.read_csv(DATA_DIR+'/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv(DATA_DIR+'/test.csv', encoding="ISO-8859-1")
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

### load product attributes ###############
df_attr = pd.read_csv(DATA_DIR+'/attributes.csv', encoding="ISO-8859-1")

print 'loading time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


### find unique brands from the attributes file
### for a few product_uids there are at least two different names in "MFG Brand Name"
### in such cases we keep only one of the names
df_all = pd.merge(df_all, df_attr[df_attr['name']=="MFG Brand Name"][['product_uid','value']], how='left', on='product_uid')
df_all['brand']=df_all['value'].fillna("").map(lambda x: x.encode('utf-8'))
df_all=df_all.drop('value',axis=1)


### Create a list of words with lowercase and uppercase letters 
### Examples: 'InSinkErator', 'EpoxyShield'
### They are words from brand names or words from product title.
### The dict is used to correct product description which contins concatenated 
### lines of text without separators : 
### ---View lawn edgings and brick/ paver edgingsUtility stakes can be used for many purposes---
### Here we need to replace 'edgingsUtility' with 'edgings utility'. 
### But we don't need to replace 'InSinkErator' with 'in sink erator'
add_space_stop_list=[]
uniq_brands=list(set(list(df_all['brand'])))
for i in range(0,len(uniq_brands)):
    uniq_brands[i]=simple_parser(uniq_brands[i])
    if re.search(r'[a-z][A-Z][a-z]',uniq_brands[i])!=None:
        for word in uniq_brands[i].split():
            if re.search(r'[a-z][A-Z][a-z]',word)!=None:
                add_space_stop_list.append(word.lower())
add_space_stop_list=list(set(add_space_stop_list))      
print len(add_space_stop_list)," words from brands in add_space_stop_list"
                
uniq_titles=list(set(list(df_all['product_title'])))
for i in range(0,len(uniq_titles)):
    uniq_titles[i]=simple_parser(uniq_titles[i])
    if re.search(r'[a-z][A-Z][a-z]',uniq_titles[i])!=None:
        for word in uniq_titles[i].split():
            if re.search(r'[a-z][A-Z][a-z]',word)!=None:
                add_space_stop_list.append(word.lower())    
add_space_stop_list=list(set(add_space_stop_list))      
print len(add_space_stop_list) ," total words from brands and product titles in add_space_stop_list\n"
                

#################################################################
##### First step of spell correction: using the Google dict
##### from the forum
# https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos

df_all['search_term']=df_all['search_term'].map(lambda x: google_dict[x] if x in google_dict.keys() else x)   




#################################################################
##### AUTOMATIC SPELL CHECKER ###################################
#################################################################

### A simple spell checker is implemented here
### First, we get unique words from search_term and product_title
### Then, we count how many times word occurs in search_term and product_title
### Finally, if the word is not present in product_title and not meaningful
### (i.e. wn.synsets(word) returns empty list), the word is likely 
### to be misspelled, so we try to correct it using bigrams, words from matched
### products or all products. The best match is chosen using 
### difflib.SequenceMatcher()


def is_word_in_string(word,s):
    return word in s.split() 
    
def create_bigrams(s):
    lst=[word for word in s.split() if len(re.sub('[^0-9]', '', word))==0 and len(word)>2]
    output=""
    i=0
    if len(lst)>=2:
        while i<len(lst)-1:
            output+= " "+lst[i]+"_"+lst[i+1]
            i+=1
    return output


df_all['product_title_simpleparsed']=df_all['product_title'].map(lambda x: simple_parser(x).lower())
df_all['search_term_simpleparsed']=df_all['search_term'].map(lambda x: simple_parser(x).lower())

str_title=" ".join(list(df_all['product_title'].map(lambda x: simple_parser(x).lower())))
str_query=" ".join(list(df_all['search_term'].map(lambda x: simple_parser(x).lower())))

# create bigrams
bigrams_str_title=" ".join(list(df_all['product_title'].map(lambda x: create_bigrams(simple_parser(x).lower()))))
bigrams_set=set(bigrams_str_title.split())

### count word frequencies for query and product title
my_dict={}
str1= str_title+" "+str_query
for word in list(set(list(str1.split()))):
    my_dict[word]={"title":0, "query":0, 'word':word}
for word in str_title.split():
    my_dict[word]["title"]+=1    
for word in str_query.split():
    my_dict[word]["query"]+=1


### 1. Process words without digits
### Potential errors: words that appear only in query
### Correct words: 5 or more times in product_title
errors_dict={}
correct_dict={}
for word in my_dict.keys():
    if len(word)>=3 and len(re.sub('[^0-9]', '', word))==0:
        if my_dict[word]["title"]==0:
            if len(wn.synsets(word))>0 \
            or (word.endswith('s') and  (word[:-1] in my_dict.keys()) and my_dict[word[:-1]]["title"]>0)\
            or (word[-1]!='s' and (word+'s' in my_dict.keys()) and my_dict[word+'s']["title"]>0):
                1
            else:
                errors_dict[word]=my_dict[word]
        elif my_dict[word]["title"]>=5:
            correct_dict[word]=my_dict[word]


### for each error word try finding a good match in bigrams, matched products, all products
cnt=0
NN=len(errors_dict.keys())
t0=time()
for i in range(0,len(errors_dict.keys())):
    word=sorted(errors_dict.keys())[i]
    cnt+=1
    lst=[]
    lst_tuple=[]
    suggested=False
    suggested_word=""
    rt_max=0
    
    # if only one word in query, use be more selective in choosing a correction
    min_query_len=min(df_all['search_term_simpleparsed'][df_all['search_term_simpleparsed'].map(lambda x: is_word_in_string(word,x))].map(lambda x: len(x.split())))
    delta=0.05*int(min_query_len<2)
    
    words_from_matched_titles=[item for item in \
        " ".join(list(set(df_all['product_title_simpleparsed'][df_all['search_term_simpleparsed'].map(lambda x: is_word_in_string(word,x))]))).split() \
        if len(item)>2 and len(re.sub('[^0-9]', '', item))==0]
    words_from_matched_titles=list(set(words_from_matched_titles))
    words_from_matched_titles.sort()
    
    source=""
    for bigram in bigrams_set:
        if bigram.replace("_","")==word:
            suggested=True