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
df_all['size_of_brands_in_product_title'] = df_all['brands_in_product_title'].map(lambda x:len(x.split(";"))).astype(np.int64)
df_all['len_of_materials_in_product_title'] = df_all['materials_in_product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_materials_in_product_title'] = df_all['materials_in_product_title'].map(lambda x:len(x.split(";"))).astype(np.int64)

df_all['len_of_brands_in_product_description'] = df_all['brands_in_product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_brands_in_product_description'] = df_all['brands_in_product_description'].map(lambda x:len(x.split(";"))).astype(np.int64)
df_all['len_of_materials_in_product_description'] = df_all['materials_in_product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_materials_in_product_description'] = df_all['materials_in_product_description'].map(lambda x:len(x.split(";"))).astype(np.int64)

df_all['len_of_brands_in_attribute_bullets'] = df_all['brands_in_attribute_bullets'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_brands_in_attribute_bullets'] = df_all['brands_in_attribute_bullets'].map(lambda x:len(x.split(";"))).astype(np.int64)
df_all['len_of_materials_in_attribute_bullets'] = df_all['materials_in_attribute_bullets'].map(lambda x:len(x.split())).astype(np.int64)
df_all['size_of_materials_in_attribute_bullets'] = df_all['materials_in_attribute_bullets'].map(lambda x:len(x.split(";"))).astype(np.int64)

df_all['len_of_query']=df_all['len_of_query_woBM']+df_all['size_of_brands_in_query']+df_all['size_of_materials_in_query']
df_all['len_of_product_title']=df_all['len_of_product_title_woBM']+df_all['size_of_brands_in_product_title']

df_all['len_of_product_description']=df_all['len_of_product_title_woBM']+df_all['size_of_brands_in_product_description']+df_all['size_of_materials_in_product_description']
### The previous line is incorrect.
### It should be 
### df_all['len_of_product_description']=df_all['len_of_product_description_woBM']+df_all['size_of_brands_in_product_description']+df_all['size_of_materials_in_product_description']

df_all['len_of_attribute_bullets']=df_all['len_of_attribute_bullets_woBM']+df_all['size_of_brands_in_attribute_bullets']+df_all['size_of_materials_in_attribute_bullets']


df_all['len_of_query_keys'] = df_all['search_term_keys_stemmed'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_product_title_keys'] = df_all['product_title_keys_stemmed'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_query_thekey'] = df_all['search_term_thekey'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_product_title_thekey'] = df_all['product_title_thekey'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_query_beforethekey'] = df_all['search_term_beforethekey'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_product_title_beforethekey'] = df_all['product_title_beforethekey'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_query_before2thekey'] = df_all['search_term_before2thekey'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_product_title_before2thekey'] = df_all['product_title_before2thekey'].map(lambda x:len(x.split())).astype(np.int64)


def find_ratio(a,b):
    if b==0:
        return 0
    else:
        return min(1,1.0*a/b)
        
for column_name in ['search_term', 'product_title', 'product_description','attribute_bullets']:
    df_all['ratio_of_nn_important_in_'+column_name]= df_all.apply(lambda x: \
        find_ratio(len(nn_important_words(x['search_term_tokens']).split()), len(x[column_name+'_parsed'])) ,axis=1)
    df_all['ratio_of_nn_unimportant_in_'+column_name]= df_all.apply(lambda x: \
        find_ratio(len(nn_unimportant_words(x['search_term_tokens']).split()), len(x[column_name+'_parsed'])) ,axis=1)
    df_all['ratio_of_vb_in_'+column_name]= df_all.apply(lambda x: \
        find_ratio(len(vb_words(x['search_term_tokens']).split()), len(x[column_name+'_parsed'])) ,axis=1)
    df_all['ratio_of_vbg_in_'+column_name]= df_all.apply(lambda x: \
        find_ratio(len(vbg_words(x['search_term_tokens']).split()), len(x[column_name+'_parsed'])) ,axis=1)
    df_all['ratio_of_jj_rb_in_'+column_name]= df_all.apply(lambda x: \
        find_ratio(len(nn_unimportant_words(x['search_term_tokens']).split()), len(x[column_name+'_parsed'])) ,axis=1)




print 'len_of_something time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


def perc_digits_in_str(s):
    if len(s.split())==0:
        output=0
    else:
        output =1.0 -1.0*len(words_wo_digits(s, minLength=1).split()) / len(s.split())
    return output

df_all['perc_digits_in_query'] = df_all['search_term_stemmed'].map(lambda x: perc_digits_in_str(x)  )
df_all['perc_digits_in_title'] = df_all['product_title_stemmed'].map(lambda x: perc_digits_in_str(x)  )
df_all['perc_digits_in_description'] = df_all['product_description_stemmed'].map(lambda x: perc_digits_in_str(x) )
df_all['perc_digits_in_bullets'] = df_all['attribute_bullets_stemmed'].map(lambda x: perc_digits_in_str(x) )

print 'perc_digits time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


#################################################################
### STEP 3: Measure whether brand and materials in query match other text
#################################################################

### the following function compares query with the corresponding product title,
### prodcut description or attribute bullets and returns:
### * number of brands/materials fully matched
### * number of brands/materials partially matched ('behr' and 'behr premium')
### * number of brands/materials with assumed match (first word of brand found, e.g. 'handy home products' and 'handy paint pail')
### * number of brands/materials in query not matched
### * convoluted output: 3 if all brands/materials fully matched,
###                      2 if all matched at least partially
###                      1 if at least one is matched but at least one is not matched
###                      0 if no brand/material in query
###                     -1 if there is brand/material in query but no brand/material in text
###                     -2 if there are brands different brands/materials in query and text 

def query_brand_material_in_attribute(str_query_brands,str_attribute_brands):
    list_query_brands=list(set(str_query_brands.split(";")))
    list_attribute_brands=list(set(str_attribute_brands.split(";")))
    while '' in list_query_brands:
        list_query_brands.remove('')
    while '' in list_attribute_brands:
        list_attribute_brands.remove('')
    
    str_attribute_brands=" ".join(str_attribute_brands.split(";"))    
    full_match=0
    partial_match=0
    assumed_match=0
    no_match=0
    num_of_query_brands=len(list_query_brands)
    num_of_attribute_brands=len(list_attribute_brands)
    if num_of_query_brands>0:
        for brand in list_query_brands:
            if brand in list_attribute_brands:
                full_match+=1
            elif ' '+brand+' ' in ' '+str_attribute_brands+' ':
                partial_match+=1
            elif (' '+brand.split()[0] in ' '+str_attribute_brands and brand.split()[0][0] not in "0123456789") or \
            (len(brand.split())>1 and (' '+brand.split()[0]+' '+brand.split()[1]) in ' '+str_attribute_brands):
                assumed_match+=1
            else:
                no_match+=1
                
    convoluted_output=0 # no brand in query
    if num_of_query_brands>0:
        if num_of_attribute_brands==0:
            convoluted_output = -1 # no brand in text, but there is brand in query
        elif no_match==0:
            if assumed_match==0:
                convoluted_output=3 # all brands fully matched
            else:
                convoluted_output=2 # all brands matched at least partially
        else:
            if full_match+ partial_match+ assumed_match>0:
                convoluted_output = 1 # one brand matched but the other is not
            else:
                convoluted_output= -2  #brand mismatched
    
    return full_match, partial_match, assumed_match, no_match, convoluted_output


df_all['brands_all']=df_all['brands_in_search_term']+"\t"+df_all['brands_in_product_title']+"\t"+df_all['brands_in_product_description']+"\t"\
+df_all['brands_in_attribute_bullets']+"\t"+df_all['brand_parsed']

df_all['query_brand_in_brand_tuple']=df_all['brands_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[4]))
df_all['query_brand_in_brand_fullmatch']=df_all['query_brand_in_brand_tuple'].map(lambda x: x[0])
df_all['query_brand_in_brand_partialmatch']=df_all['query_brand_in_brand_tuple'].map(lambda x: x[1])
df_all['query_brand_in_brand_assumedmatch']=df_all['query_brand_in_brand_tuple'].map(lambda x: x[2])
df_all['query_brand_in_brand_nomatch']=df_all['query_brand_in_brand_tuple'].map(lambda x: x[3])
df_all['query_brand_in_brand_convoluted']=df_all['query_brand_in_brand_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_brand_in_brand_tuple'],axis=1)

df_all['query_brand_in_all_tuple']=df_all['brands_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],";".join(x.split("\t")[1:])))
df_all['query_brand_in_all_fullmatch']=df_all['query_brand_in_all_tuple'].map(lambda x: x[0])
df_all['query_brand_in_all_partialmatch']=df_all['query_brand_in_all_tuple'].map(lambda x: x[1])
df_all['query_brand_in_all_assumedmatch']=df_all['query_brand_in_all_tuple'].map(lambda x: x[2])
df_all['query_brand_in_all_nomatch']=df_all['query_brand_in_all_tuple'].map(lambda x: x[3])
df_all['query_brand_in_all_convoluted']=df_all['query_brand_in_all_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_brand_in_all_tuple'],axis=1)

df_all['query_brand_in_title_tuple']=df_all['brands_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[1]))
df_all['query_brand_in_title_fullmatch']=df_all['query_brand_in_title_tuple'].map(lambda x: x[0])
df_all['query_brand_in_title_partialmatch']=df_all['query_brand_in_title_tuple'].map(lambda x: x[1])
df_all['query_brand_in_title_assumedmatch']=df_all['query_brand_in_title_tuple'].map(lambda x: x[2])
df_all['query_brand_in_title_nomatch']=df_all['query_brand_in_title_tuple'].map(lambda x: x[3])
df_all['query_brand_in_title_convoluted']=df_all['query_brand_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_brand_in_title_tuple'],axis=1)

df_all['query_brand_in_description_tuple']=df_all['brands_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[2]))
df_all['query_brand_in_description_convoluted']=df_all['query_brand_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_brand_in_description_tuple'],axis=1)

df_all['query_brand_in_bullets_tuple']=df_all['brands_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[3]))
df_all['query_brand_in_bullets_convoluted']=df_all['query_brand_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_brand_in_bullets_tuple'],axis=1)

df_all=df_all.drop(['brands_all'],axis=1)

print 'create brand match variables time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()



########################################

df_all['materials_all']=df_all['materials_in_search_term']+"\t"+df_all['materials_in_product_title']+"\t"+df_all['materials_in_product_description']+"\t"\
+df_all['materials_in_attribute_bullets']+"\t"+df_all['material_parsed']

df_all['query_material_in_material_tuple']=df_all['materials_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[4]))
df_all['query_material_in_material_fullmatch']=df_all['query_material_in_material_tuple'].map(lambda x: x[0])
df_all['query_material_in_material_partialmatch']=df_all['query_material_in_material_tuple'].map(lambda x: x[1])
df_all['query_material_in_material_assumedmatch']=df_all['query_material_in_material_tuple'].map(lambda x: x[2])
df_all['query_material_in_material_nomatch']=df_all['query_material_in_material_tuple'].map(lambda x: x[3])
df_all['query_material_in_material_convoluted']=df_all['query_material_in_material_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_material_in_material_tuple'],axis=1)

df_all['query_material_in_all_tuple']=df_all['materials_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],";".join(x.split("\t")[1:])))
df_all['query_material_in_all_fullmatch']=df_all['query_material_in_all_tuple'].map(lambda x: x[0])
df_all['query_material_in_all_partialmatch']=df_all['query_material_in_all_tuple'].map(lambda x: x[1])
df_all['query_material_in_all_assumedmatch']=df_all['query_material_in_all_tuple'].map(lambda x: x[2])
df_all['query_material_in_all_nomatch']=df_all['query_material_in_all_tuple'].map(lambda x: x[3])
df_all['query_material_in_all_convoluted']=df_all['query_material_in_all_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_material_in_all_tuple'],axis=1)

df_all['query_material_in_title_tuple']=df_all['materials_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[1]))
df_all['query_material_in_title_convoluted']=df_all['query_material_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_material_in_title_tuple'],axis=1)

df_all['query_material_in_description_tuple']=df_all['materials_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[2]))
df_all['query_material_in_description_convoluted']=df_all['query_material_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_material_in_description_tuple'],axis=1)

df_all['query_material_in_bullets_tuple']=df_all['materials_all'].map(lambda x: query_brand_material_in_attribute(x.split("\t")[0],x.split("\t")[3]))
df_all['query_material_in_bullets_convoluted']=df_all['query_material_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['query_material_in_bullets_tuple'],axis=1)

df_all=df_all.drop(['materials_all'],axis=1)

print 'create material match variables time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()





#################################################################
### STEP 4: Count number of similar words/letters/bigrams etc
#################################################################

################################
### query vs product title


df_all['wordFor_in_title_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_for_stemmed'],x['product_title_stemmed'],string_only=True),axis=1)
df_all['wordFor_in_title_string_only_num'] = df_all['wordFor_in_title_string_only_tuple'].map(lambda x: x[0])
df_all['wordFor_in_title_string_only_let'] = df_all['wordFor_in_title_string_only_tuple'].map(lambda x: x[2])
df_all['wordFor_in_title_string_only_letratio'] = df_all['wordFor_in_title_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['wordFor_in_title_string_only_tuple'],axis=1)

df_all['wordWith_in_title_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_with_stemmed'],x['product_title_stemmed'],string_only=True),axis=1)
df_all['wordWith_in_title_string_only_num'] = df_all['wordWith_in_title_string_only_tuple'].map(lambda x: x[0])
df_all['wordWith_in_title_string_only_let'] = df_all['wordWith_in_title_string_only_tuple'].map(lambda x: x[2])
df_all['wordWith_in_title_string_only_letratio'] = df_all['wordWith_in_title_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['wordWith_in_title_string_only_tuple'],axis=1)

df_all['prtitleWithout_in_query_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['product_title_without_stemmed'],x['search_term_stemmed'],string_only=True),axis=1)
df_all['prtitleWithout_in_query_string_only_num'] = df_all['prtitleWithout_in_query_string_only_tuple'].map(lambda x: x[0])
df_all['prtitleWithout_in_query_string_only_let'] = df_all['prtitleWithout_in_query_string_only_tuple'].map(lambda x: x[2])
df_all['prtitleWithout_in_query_string_only_letratio'] = df_all['prtitleWithout_in_query_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['prtitleWithout_in_query_string_only_tuple'],axis=1)

df_all['query_in_title']=df_all.apply(lambda x: \
            query_in_text(x['search_term_stemmed'],x['product_title_stemmed']),axis=1)

df_all['word_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_stemmed'],x['product_title_stemmed']),axis=1)
df_all['word_in_title_num'] = df_all['word_in_title_tuple'].map(lambda x: x[0])
df_all['word_in_title_sum'] = df_all['word_in_title_tuple'].map(lambda x: x[1])
df_all['word_in_title_let'] = df_all['word_in_title_tuple'].map(lambda x: x[2])
df_all['word_in_title_numratio'] = df_all['word_in_title_tuple'].map(lambda x: x[3])
df_all['word_in_title_letratio'] = df_all['word_in_title_tuple'].map(lambda x: x[4])
df_all['word_in_title_string'] = df_all['word_in_title_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['word_in_title_tuple'],axis=1)

df_all['word_in_title_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_stemmed'],x['product_title_stemmed'],string_only=True),axis=1)
df_all['word_in_title_string_only_num'] = df_all['word_in_title_string_only_tuple'].map(lambda x: x[0])
df_all['word_in_title_string_only_sum'] = df_all['word_in_title_string_only_tuple'].map(lambda x: x[1])
df_all['word_in_title_string_only_let'] = df_all['word_in_title_string_only_tuple'].map(lambda x: x[2])
df_all['word_in_title_string_only_numratio'] = df_all['word_in_title_string_only_tuple'].map(lambda x: x[3])
df_all['word_in_title_string_only_letratio'] = df_all['word_in_title_string_only_tuple'].map(lambda x: x[4])
df_all['word_in_title_string_only_string'] = df_all['word_in_title_string_only_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['word_in_title_string_only_tuple'],axis=1)

df_all['word_in_title_w_dash']=df_all.apply(lambda x: \
            str_common_word(words_w_dash(x['search_term_stemmed']),words_w_dash(x['product_title_stemmed']))[0],axis=1)

df_all['two_words_in_title_tuple']=df_all.apply(lambda x: \
            str_2common_words(x['search_term_stemmed'],x['product_title_stemmed']),axis=1)
df_all['two_words_in_title_num'] = df_all['two_words_in_title_tuple'].map(lambda x: x[0])
df_all['two_words_in_title_sum'] = df_all['two_words_in_title_tuple'].map(lambda x: x[1])
df_all['two_words_in_title_let'] = df_all['two_words_in_title_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['two_words_in_title_tuple'],axis=1)

df_all['two_words_in_title_string_only_tuple']=df_all.apply(lambda x: \
            str_2common_words(x['search_term_stemmed'],x['product_title_stemmed'],string_only=True),axis=1)
df_all['two_words_in_title_string_only_num'] = df_all['two_words_in_title_string_only_tuple'].map(lambda x: x[0])
df_all['two_words_in_title_string_only_sum'] = df_all['two_words_in_title_string_only_tuple'].map(lambda x: x[1])
df_all['two_words_in_title_string_only_let'] = df_all['two_words_in_title_string_only_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['two_words_in_title_string_only_tuple'],axis=1)

df_all['common_digits_in_title_tuple']=df_all.apply(lambda x: \
            str_common_digits(x['search_term_stemmed'],x['product_title_stemmed']),axis=1)
df_all['len_of_digits_in_query'] = df_all['common_digits_in_title_tuple'].map(lambda x: x[0])
df_all['len_of_digits_in_title'] = df_all['common_digits_in_title_tuple'].map(lambda x: x[1])
df_all['common_digits_in_title_num'] = df_all['common_digits_in_title_tuple'].map(lambda x: x[2])
df_all['common_digits_in_title_ratio'] = df_all['common_digits_in_title_tuple'].map(lambda x: x[3])
df_all['common_digits_in_title_jaccard'] = df_all['common_digits_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['common_digits_in_title_tuple'],axis=1)

df_all['nn_important_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['product_title_stemmed']),axis=1)
df_all['nn_important_in_title_num'] = df_all['nn_important_in_title_tuple'].map(lambda x: x[0])
df_all['nn_important_in_title_sum'] = df_all['nn_important_in_title_tuple'].map(lambda x: x[1])
df_all['nn_important_in_title_let'] = df_all['nn_important_in_title_tuple'].map(lambda x: x[2])
df_all['nn_important_in_title_numratio'] = df_all['nn_important_in_title_tuple'].map(lambda x: x[3])
df_all['nn_important_in_title_letratio'] = df_all['nn_important_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_title_tuple'],axis=1)

df_all['nn_unimportant_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_unimportant_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['product_title_stemmed']),axis=1)
df_all['nn_unimportant_in_title_num'] = df_all['nn_unimportant_in_title_tuple'].map(lambda x: x[0])
df_all['nn_unimportant_in_title_let'] = df_all['nn_unimportant_in_title_tuple'].map(lambda x: x[2])
df_all['nn_unimportant_in_title_letratio'] = df_all['nn_unimportant_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_unimportant_in_title_tuple'],axis=1)

"""
df_all['vbg_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(vbg_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['product_title_stemmed']),axis=1)
df_all['vbg_in_title_num'] = df_all['vbg_in_title_tuple'].map(lambda x: x[0])
df_all['vbg_in_title_sum'] = df_all['vbg_in_title_tuple'].map(lambda x: x[1])
df_all['vbg_in_title_let'] = df_all['vbg_in_title_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['vbg_in_title_tuple'],axis=1)

df_all['jj_rb_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(jj_rb_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['product_title_stemmed']),axis=1)
df_all['jj_rb_in_title_num'] = df_all['jj_rb_in_title_tuple'].map(lambda x: x[0])
df_all['jj_rb_in_title_sum'] = df_all['jj_rb_in_title_tuple'].map(lambda x: x[1])
df_all['jj_rb_in_title_let'] = df_all['jj_rb_in_title_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['jj_rb_in_title_tuple'],axis=1)
"""

df_all['nn_important_in_nn_important_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_important_words(x['product_title_tokens']),stoplist=stoplist_wo_can)),axis=1)
df_all['nn_important_in_nn_important_in_title_num'] = df_all['nn_important_in_nn_important_in_title_tuple'].map(lambda x: x[0])
df_all['nn_important_in_nn_important_in_title_sum'] = df_all['nn_important_in_nn_important_in_title_tuple'].map(lambda x: x[1])
df_all['nn_important_in_nn_important_in_title_let'] = df_all['nn_important_in_nn_important_in_title_tuple'].map(lambda x: x[2])
df_all['nn_important_in_nn_important_in_title_letratio'] = df_all['nn_important_in_nn_important_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_nn_important_in_title_tuple'],axis=1)

df_all['nn_important_in_nn_unimportant_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_unimportant_words(x['product_title_tokens']),stoplist=stoplist_wo_can)),axis=1)
df_all['nn_important_in_nn_unimportant_in_title_num'] = df_all['nn_important_in_nn_unimportant_in_title_tuple'].map(lambda x: x[0])
df_all['nn_important_in_nn_unimportant_in_title_sum'] = df_all['nn_important_in_nn_unimportant_in_title_tuple'].map(lambda x: x[1])
df_all['nn_important_in_nn_unimportant_in_title_let'] = df_all['nn_important_in_nn_unimportant_in_title_tuple'].map(lambda x: x[2])
df_all['nn_important_in_nn_unimportant_in_title_letratio'] = df_all['nn_important_in_nn_unimportant_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_nn_unimportant_in_title_tuple'],axis=1)

df_all['nn_unimportant_in_nn_important_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_unimportant_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_important_words(x['product_title_tokens']),stoplist=stoplist_wo_can)),axis=1)
df_all['nn_unimportant_in_nn_important_in_title_num'] = df_all['nn_unimportant_in_nn_important_in_title_tuple'].map(lambda x: x[0])
df_all['nn_unimportant_in_nn_important_in_title_sum'] = df_all['nn_unimportant_in_nn_important_in_title_tuple'].map(lambda x: x[1])
df_all['nn_unimportant_in_nn_important_in_title_let'] = df_all['nn_unimportant_in_nn_important_in_title_tuple'].map(lambda x: x[2])
df_all['nn_unimportant_in_nn_important_in_title_letratio'] = df_all['nn_unimportant_in_nn_important_in_title_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_unimportant_in_nn_important_in_title_tuple'],axis=1)

df_all['jj_rb_in_jj_rb_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(jj_rb_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(jj_rb_words(x['product_title_tokens']),stoplist=stoplist_wo_can)),axis=1)
df_all['jj_rb_in_jj_rb_in_title_num'] = df_all['jj_rb_in_jj_rb_in_title_tuple'].map(lambda x: x[0])
df_all['jj_rb_in_jj_rb_in_title_sum'] = df_all['jj_rb_in_jj_rb_in_title_tuple'].map(lambda x: x[1])
df_all['jj_rb_in_jj_rb_in_title_let'] = df_all['jj_rb_in_jj_rb_in_title_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['jj_rb_in_jj_rb_in_title_tuple'],axis=1)

df_all['vbg_in_vbg_in_title_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(vbg_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(vbg_words(x['product_title_tokens']),stoplist=stoplist_wo_can)),axis=1)
df_all['vbg_in_vbg_in_title_num'] = df_all['vbg_in_vbg_in_title_tuple'].map(lambda x: x[0])
df_all['vbg_in_vbg_in_title_sum'] = df_all['vbg_in_vbg_in_title_tuple'].map(lambda x: x[1])
df_all['vbg_in_vbg_in_title_let'] = df_all['vbg_in_vbg_in_title_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['vbg_in_vbg_in_title_tuple'],axis=1)
print 'words_in_title time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()


###################################
###################################

################################
### query vs product description


df_all['wordFor_in_description_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_for_stemmed'],x['product_description_stemmed'],string_only=True),axis=1)
df_all['wordFor_in_description_string_only_num'] = df_all['wordFor_in_description_string_only_tuple'].map(lambda x: x[0])
df_all['wordFor_in_description_string_only_let'] = df_all['wordFor_in_description_string_only_tuple'].map(lambda x: x[2])
df_all['wordFor_in_description_string_only_letratio'] = df_all['wordFor_in_description_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['wordFor_in_description_string_only_tuple'],axis=1)

df_all['wordWith_in_description_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_with_stemmed'],x['product_description_stemmed'],string_only=True),axis=1)
df_all['wordWith_in_description_string_only_num'] = df_all['wordWith_in_description_string_only_tuple'].map(lambda x: x[0])
df_all['wordWith_in_description_string_only_let'] = df_all['wordWith_in_description_string_only_tuple'].map(lambda x: x[2])
df_all['wordWith_in_description_string_only_letratio'] = df_all['wordWith_in_description_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['wordWith_in_description_string_only_tuple'],axis=1)

df_all['query_in_description']=df_all.apply(lambda x: \
            query_in_text(x['search_term_stemmed'],x['product_description_stemmed']),axis=1)

df_all['word_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_stemmed'],x['product_description_stemmed']),axis=1)
df_all['word_in_description_num'] = df_all['word_in_description_tuple'].map(lambda x: x[0])
df_all['word_in_description_sum'] = df_all['word_in_description_tuple'].map(lambda x: x[1])
df_all['word_in_description_let'] = df_all['word_in_description_tuple'].map(lambda x: x[2])
df_all['word_in_description_numratio'] = df_all['word_in_description_tuple'].map(lambda x: x[3])
df_all['word_in_description_letratio'] = df_all['word_in_description_tuple'].map(lambda x: x[4])
df_all['word_in_description_string'] = df_all['word_in_description_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['word_in_description_tuple'],axis=1)

df_all['word_in_description_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_stemmed'],x['product_description_stemmed'],string_only=True),axis=1)
df_all['word_in_description_string_only_num'] = df_all['word_in_description_string_only_tuple'].map(lambda x: x[0])
df_all['word_in_description_string_only_sum'] = df_all['word_in_description_string_only_tuple'].map(lambda x: x[1])
df_all['word_in_description_string_only_let'] = df_all['word_in_description_string_only_tuple'].map(lambda x: x[2])
df_all['word_in_description_string_only_numratio'] = df_all['word_in_description_string_only_tuple'].map(lambda x: x[3])
df_all['word_in_description_string_only_letratio'] = df_all['word_in_description_string_only_tuple'].map(lambda x: x[4])
df_all['word_in_description_string_only_string'] = df_all['word_in_description_string_only_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['word_in_description_string_only_tuple'],axis=1)

df_all['word_in_description_w_dash']=df_all.apply(lambda x: \
            str_common_word(words_w_dash(x['search_term_stemmed']),words_w_dash(x['product_description_stemmed']))[0],axis=1)

df_all['two_words_in_description_tuple']=df_all.apply(lambda x: \
            str_2common_words(x['search_term_stemmed'],x['product_description_stemmed']),axis=1)
df_all['two_words_in_description_num'] = df_all['two_words_in_description_tuple'].map(lambda x: x[0])
df_all['two_words_in_description_sum'] = df_all['two_words_in_description_tuple'].map(lambda x: x[1])
df_all['two_words_in_description_let'] = df_all['two_words_in_description_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['two_words_in_description_tuple'],axis=1)

df_all['two_words_in_description_string_only_tuple']=df_all.apply(lambda x: \
            str_2common_words(x['search_term_stemmed'],x['product_description_stemmed'],string_only=True),axis=1)
df_all['two_words_in_description_string_only_num'] = df_all['two_words_in_description_string_only_tuple'].map(lambda x: x[0])
df_all['two_words_in_description_string_only_sum'] = df_all['two_words_in_description_string_only_tuple'].map(lambda x: x[1])
df_all['two_words_in_description_string_only_let'] = df_all['two_words_in_description_string_only_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['two_words_in_description_string_only_tuple'],axis=1)

df_all['common_digits_in_description_tuple']=df_all.apply(lambda x: \
            str_common_digits(x['search_term_stemmed'],x['product_description_stemmed']),axis=1)
df_all['len_of_digits_in_query'] = df_all['common_digits_in_description_tuple'].map(lambda x: x[0])
df_all['len_of_digits_in_description'] = df_all['common_digits_in_description_tuple'].map(lambda x: x[1])
df_all['common_digits_in_description_num'] = df_all['common_digits_in_description_tuple'].map(lambda x: x[2])
df_all['common_digits_in_description_ratio'] = df_all['common_digits_in_description_tuple'].map(lambda x: x[3])
df_all['common_digits_in_description_jaccard'] = df_all['common_digits_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['common_digits_in_description_tuple'],axis=1)

df_all['nn_important_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['product_description_stemmed']),axis=1)
df_all['nn_important_in_description_num'] = df_all['nn_important_in_description_tuple'].map(lambda x: x[0])
df_all['nn_important_in_description_sum'] = df_all['nn_important_in_description_tuple'].map(lambda x: x[1])
df_all['nn_important_in_description_let'] = df_all['nn_important_in_description_tuple'].map(lambda x: x[2])
df_all['nn_important_in_description_numratio'] = df_all['nn_important_in_description_tuple'].map(lambda x: x[3])
df_all['nn_important_in_description_letratio'] = df_all['nn_important_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_description_tuple'],axis=1)

df_all['nn_unimportant_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_unimportant_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['product_description_stemmed']),axis=1)
df_all['nn_unimportant_in_description_num'] = df_all['nn_unimportant_in_description_tuple'].map(lambda x: x[0])
df_all['nn_unimportant_in_description_let'] = df_all['nn_unimportant_in_description_tuple'].map(lambda x: x[2])
df_all['nn_unimportant_in_description_letratio'] = df_all['nn_unimportant_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_unimportant_in_description_tuple'],axis=1)


df_all['nn_important_in_nn_important_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_important_words(x['product_description_tokens']))),axis=1)
df_all['nn_important_in_nn_important_in_description_num'] = df_all['nn_important_in_nn_important_in_description_tuple'].map(lambda x: x[0])
df_all['nn_important_in_nn_important_in_description_sum'] = df_all['nn_important_in_nn_important_in_description_tuple'].map(lambda x: x[1])
df_all['nn_important_in_nn_important_in_description_let'] = df_all['nn_important_in_nn_important_in_description_tuple'].map(lambda x: x[2])
df_all['nn_important_in_nn_important_in_description_letratio'] = df_all['nn_important_in_nn_important_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_nn_important_in_description_tuple'],axis=1)

df_all['nn_important_in_nn_unimportant_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_unimportant_words(x['product_description_tokens']))),axis=1)
df_all['nn_important_in_nn_unimportant_in_description_num'] = df_all['nn_important_in_nn_unimportant_in_description_tuple'].map(lambda x: x[0])
df_all['nn_important_in_nn_unimportant_in_description_sum'] = df_all['nn_important_in_nn_unimportant_in_description_tuple'].map(lambda x: x[1])
df_all['nn_important_in_nn_unimportant_in_description_let'] = df_all['nn_important_in_nn_unimportant_in_description_tuple'].map(lambda x: x[2])
df_all['nn_important_in_nn_unimportant_in_description_letratio'] = df_all['nn_important_in_nn_unimportant_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_nn_unimportant_in_description_tuple'],axis=1)

df_all['nn_unimportant_in_nn_important_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_unimportant_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_important_words(x['product_description_tokens']))),axis=1)
df_all['nn_unimportant_in_nn_important_in_description_num'] = df_all['nn_unimportant_in_nn_important_in_description_tuple'].map(lambda x: x[0])
df_all['nn_unimportant_in_nn_important_in_description_sum'] = df_all['nn_unimportant_in_nn_important_in_description_tuple'].map(lambda x: x[1])
df_all['nn_unimportant_in_nn_important_in_description_let'] = df_all['nn_unimportant_in_nn_important_in_description_tuple'].map(lambda x: x[2])
df_all['nn_unimportant_in_nn_important_in_description_letratio'] = df_all['nn_unimportant_in_nn_important_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_unimportant_in_nn_important_in_description_tuple'],axis=1)

df_all['jj_rb_in_jj_rb_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(jj_rb_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(jj_rb_words(x['product_description_tokens']))),axis=1)
df_all['jj_rb_in_jj_rb_in_description_num'] = df_all['jj_rb_in_jj_rb_in_description_tuple'].map(lambda x: x[0])
df_all['jj_rb_in_jj_rb_in_description_sum'] = df_all['jj_rb_in_jj_rb_in_description_tuple'].map(lambda x: x[1])
df_all['jj_rb_in_jj_rb_in_description_let'] = df_all['jj_rb_in_jj_rb_in_description_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['jj_rb_in_jj_rb_in_description_tuple'],axis=1)

df_all['vbg_in_vbg_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(vbg_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(vbg_words(x['product_description_tokens']))),axis=1)
df_all['vbg_in_vbg_in_description_num'] = df_all['vbg_in_vbg_in_description_tuple'].map(lambda x: x[0])
df_all['vbg_in_vbg_in_description_sum'] = df_all['vbg_in_vbg_in_description_tuple'].map(lambda x: x[1])
df_all['vbg_in_vbg_in_description_let'] = df_all['vbg_in_vbg_in_description_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['vbg_in_vbg_in_description_tuple'],axis=1)
print 'words_in_description time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()




###################################
###################################


################################
### query vs attribute bullets

df_all['wordFor_in_bullets_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_for_stemmed'],x['attribute_bullets_stemmed'],string_only=True),axis=1)
df_all['wordFor_in_bullets_string_only_num'] = df_all['wordFor_in_bullets_string_only_tuple'].map(lambda x: x[0])
df_all['wordFor_in_bullets_string_only_let'] = df_all['wordFor_in_bullets_string_only_tuple'].map(lambda x: x[2])
df_all['wordFor_in_bullets_string_only_letratio'] = df_all['wordFor_in_bullets_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['wordFor_in_bullets_string_only_tuple'],axis=1)

df_all['wordWith_in_bullets_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_with_stemmed'],x['attribute_bullets_stemmed'],string_only=True),axis=1)
df_all['wordWith_in_bullets_string_only_num'] = df_all['wordWith_in_bullets_string_only_tuple'].map(lambda x: x[0])
df_all['wordWith_in_bullets_string_only_let'] = df_all['wordWith_in_bullets_string_only_tuple'].map(lambda x: x[2])
df_all['wordWith_in_bullets_string_only_letratio'] = df_all['wordWith_in_bullets_string_only_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['wordWith_in_bullets_string_only_tuple'],axis=1)

df_all['query_in_bullets']=df_all.apply(lambda x: \
            query_in_text(x['search_term_stemmed'],x['attribute_bullets_stemmed']),axis=1)

df_all['word_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_stemmed'],x['attribute_bullets_stemmed']),axis=1)
df_all['word_in_bullets_num'] = df_all['word_in_bullets_tuple'].map(lambda x: x[0])
df_all['word_in_bullets_sum'] = df_all['word_in_bullets_tuple'].map(lambda x: x[1])
df_all['word_in_bullets_let'] = df_all['word_in_bullets_tuple'].map(lambda x: x[2])
df_all['word_in_bullets_numratio'] = df_all['word_in_bullets_tuple'].map(lambda x: x[3])
df_all['word_in_bullets_letratio'] = df_all['word_in_bullets_tuple'].map(lambda x: x[4])
df_all['word_in_bullets_string'] = df_all['word_in_bullets_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['word_in_bullets_tuple'],axis=1)

df_all['word_in_bullets_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_stemmed'],x['attribute_bullets_stemmed'],string_only=True),axis=1)
df_all['word_in_bullets_string_only_num'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[0])
df_all['word_in_bullets_string_only_sum'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[1])
df_all['word_in_bullets_string_only_let'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[2])
df_all['word_in_bullets_string_only_numratio'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[3])
df_all['word_in_bullets_string_only_letratio'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[4])
df_all['word_in_bullets_string_only_string'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[5])
df_all=df_all.drop(['word_in_bullets_string_only_tuple'],axis=1)

df_all['word_in_bullets_w_dash']=df_all.apply(lambda x: \
            str_common_word(words_w_dash(x['search_term_stemmed']),words_w_dash(x['attribute_bullets_stemmed']))[0],axis=1)

df_all['two_words_in_bullets_tuple']=df_all.apply(lambda x: \
            str_2common_words(x['search_term_stemmed'],x['attribute_bullets_stemmed']),axis=1)
df_all['two_words_in_bullets_num'] = df_all['two_words_in_bullets_tuple'].map(lambda x: x[0])
df_all['two_words_in_bullets_sum'] = df_all['two_words_in_bullets_tuple'].map(lambda x: x[1])
df_all['two_words_in_bullets_let'] = df_all['two_words_in_bullets_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['two_words_in_bullets_tuple'],axis=1)

df_all['two_words_in_bullets_string_only_tuple']=df_all.apply(lambda x: \
            str_2common_words(x['search_term_stemmed'],x['attribute_bullets_stemmed'],string_only=True),axis=1)
df_all['two_words_in_bullets_string_only_num'] = df_all['two_words_in_bullets_string_only_tuple'].map(lambda x: x[0])
df_all['two_words_in_bullets_string_only_sum'] = df_all['two_words_in_bullets_string_only_tuple'].map(lambda x: x[1])
df_all['two_words_in_bullets_string_only_let'] = df_all['two_words_in_bullets_string_only_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['two_words_in_bullets_string_only_tuple'],axis=1)

df_all['common_digits_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_digits(x['search_term_stemmed'],x['attribute_bullets_stemmed']),axis=1)
df_all['len_of_digits_in_query'] = df_all['common_digits_in_bullets_tuple'].map(lambda x: x[0])
df_all['len_of_digits_in_bullets'] = df_all['common_digits_in_bullets_tuple'].map(lambda x: x[1])
df_all['common_digits_in_bullets_num'] = df_all['common_digits_in_bullets_tuple'].map(lambda x: x[2])
df_all['common_digits_in_bullets_ratio'] = df_all['common_digits_in_bullets_tuple'].map(lambda x: x[3])
df_all['common_digits_in_bullets_jaccard'] = df_all['common_digits_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['common_digits_in_bullets_tuple'],axis=1)

df_all['nn_important_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['attribute_bullets_stemmed']),axis=1)
df_all['nn_important_in_bullets_num'] = df_all['nn_important_in_bullets_tuple'].map(lambda x: x[0])
df_all['nn_important_in_bullets_sum'] = df_all['nn_important_in_bullets_tuple'].map(lambda x: x[1])
df_all['nn_important_in_bullets_let'] = df_all['nn_important_in_bullets_tuple'].map(lambda x: x[2])
df_all['nn_important_in_bullets_numratio'] = df_all['nn_important_in_bullets_tuple'].map(lambda x: x[3])
df_all['nn_important_in_bullets_letratio'] = df_all['nn_important_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_bullets_tuple'],axis=1)

df_all['nn_unimportant_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_unimportant_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            x['attribute_bullets_stemmed']),axis=1)
df_all['nn_unimportant_in_bullets_num'] = df_all['nn_unimportant_in_bullets_tuple'].map(lambda x: x[0])
df_all['nn_unimportant_in_bullets_let'] = df_all['nn_unimportant_in_bullets_tuple'].map(lambda x: x[2])
df_all['nn_unimportant_in_bullets_letratio'] = df_all['nn_unimportant_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_unimportant_in_bullets_tuple'],axis=1)


df_all['nn_important_in_nn_important_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_important_words(x['attribute_bullets_tokens']))),axis=1)
df_all['nn_important_in_nn_important_in_bullets_num'] = df_all['nn_important_in_nn_important_in_bullets_tuple'].map(lambda x: x[0])
df_all['nn_important_in_nn_important_in_bullets_sum'] = df_all['nn_important_in_nn_important_in_bullets_tuple'].map(lambda x: x[1])
df_all['nn_important_in_nn_important_in_bullets_let'] = df_all['nn_important_in_nn_important_in_bullets_tuple'].map(lambda x: x[2])
df_all['nn_important_in_nn_important_in_bullets_letratio'] = df_all['nn_important_in_nn_important_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_nn_important_in_bullets_tuple'],axis=1)

df_all['nn_important_in_nn_unimportant_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_important_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_unimportant_words(x['attribute_bullets_tokens']))),axis=1)
df_all['nn_important_in_nn_unimportant_in_bullets_num'] = df_all['nn_important_in_nn_unimportant_in_bullets_tuple'].map(lambda x: x[0])
df_all['nn_important_in_nn_unimportant_in_bullets_sum'] = df_all['nn_important_in_nn_unimportant_in_bullets_tuple'].map(lambda x: x[1])
df_all['nn_important_in_nn_unimportant_in_bullets_let'] = df_all['nn_important_in_nn_unimportant_in_bullets_tuple'].map(lambda x: x[2])
df_all['nn_important_in_nn_unimportant_in_bullets_letratio'] = df_all['nn_important_in_nn_unimportant_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_important_in_nn_unimportant_in_bullets_tuple'],axis=1)

df_all['nn_unimportant_in_nn_important_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(nn_unimportant_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(nn_important_words(x['attribute_bullets_tokens']))),axis=1)
df_all['nn_unimportant_in_nn_important_in_bullets_num'] = df_all['nn_unimportant_in_nn_important_in_bullets_tuple'].map(lambda x: x[0])
df_all['nn_unimportant_in_nn_important_in_bullets_sum'] = df_all['nn_unimportant_in_nn_important_in_bullets_tuple'].map(lambda x: x[1])
df_all['nn_unimportant_in_nn_important_in_bullets_let'] = df_all['nn_unimportant_in_nn_important_in_bullets_tuple'].map(lambda x: x[2])
df_all['nn_unimportant_in_nn_important_in_bullets_letratio'] = df_all['nn_unimportant_in_nn_important_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['nn_unimportant_in_nn_important_in_bullets_tuple'],axis=1)

df_all['jj_rb_in_jj_rb_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(jj_rb_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(jj_rb_words(x['attribute_bullets_tokens']))),axis=1)
df_all['jj_rb_in_jj_rb_in_bullets_num'] = df_all['jj_rb_in_jj_rb_in_bullets_tuple'].map(lambda x: x[0])
df_all['jj_rb_in_jj_rb_in_bullets_sum'] = df_all['jj_rb_in_jj_rb_in_bullets_tuple'].map(lambda x: x[1])
df_all['jj_rb_in_jj_rb_in_bullets_let'] = df_all['jj_rb_in_jj_rb_in_bullets_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['jj_rb_in_jj_rb_in_bullets_tuple'],axis=1)

df_all['vbg_in_vbg_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(str_stemmer_wo_parser(vbg_words(x['search_term_tokens']),stoplist=stoplist_wo_can),\
                            str_stemmer_wo_parser(vbg_words(x['attribute_bullets_tokens']))),axis=1)
df_all['vbg_in_vbg_in_bullets_num'] = df_all['vbg_in_vbg_in_bullets_tuple'].map(lambda x: x[0])
df_all['vbg_in_vbg_in_bullets_sum'] = df_all['vbg_in_vbg_in_bullets_tuple'].map(lambda x: x[1])
df_all['vbg_in_vbg_in_bullets_let'] = df_all['vbg_in_vbg_in_bullets_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['vbg_in_vbg_in_bullets_tuple'],axis=1)
print 'words_in_bullets time:',round((time()-t0)/60,1) ,'minutes\n'
t0 = time()

















#################################################################
### STEP 5: Process the most important words (keywords)
#################################################################



############################################################
### count common words/letters and jaccard coefficients


t0 = time()

df_all['keyword_in_titlekeys_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_keys_stemmed'],x['product_title_keys_stemmed']),axis=1)
df_all['keyword_in_titlekeys_num'] = df_all['keyword_in_titlekeys_tuple'].map(lambda x: x[0])
df_all['keyword_in_titlekeys_sum'] = df_all['keyword_in_titlekeys_tuple'].map(lambda x: x[1])
df_all['keyword_in_titlekeys_let'] = df_all['keyword_in_titlekeys_tuple'].map(lambda x: x[2])
df_all['keyword_in_titlekeys_numratio'] = df_all['keyword_in_titlekeys_tuple'].map(lambda x: x[3])
df_all['keyword_in_titlekeys_letratio'] = df_all['keyword_in_titlekeys_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['keyword_in_titlekeys_tuple'],axis=1)

df_all['keyword_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_keys_stemmed'],x['product_description_stemmed']),axis=1)
df_all['keyword_in_description_num'] = df_all['keyword_in_description_tuple'].map(lambda x: x[0])
df_all['keyword_in_description_sum'] = df_all['keyword_in_description_tuple'].map(lambda x: x[1])
df_all['keyword_in_description_let'] = df_all['keyword_in_description_tuple'].map(lambda x: x[2])
df_all['keyword_in_description_numratio'] = df_all['keyword_in_description_tuple'].map(lambda x: x[3])
df_all['keyword_in_description_letratio'] = df_all['keyword_in_description_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['keyword_in_description_tuple'],axis=1)

df_all['keyword_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_keys_stemmed'],x['attribute_bullets_stemmed']),axis=1)
df_all['keyword_in_bullets_num'] = df_all['keyword_in_bullets_tuple'].map(lambda x: x[0])
df_all['keyword_in_bullets_sum'] = df_all['keyword_in_bullets_tuple'].map(lambda x: x[1])
df_all['keyword_in_bullets_let'] = df_all['keyword_in_bullets_tuple'].map(lambda x: x[2])
df_all['keyword_in_bullets_numratio'] = df_all['keyword_in_bullets_tuple'].map(lambda x: x[3])
df_all['keyword_in_bullets_letratio'] = df_all['keyword_in_bullets_tuple'].map(lambda x: x[4])
df_all=df_all.drop(['keyword_in_bullets_tuple'],axis=1)


df_all['jaccard_keyword_in_titlekeys_tuple']=df_all.apply(lambda x: \
            str_jaccard(x['search_term_keys_stemmed'],x['product_title_keys_stemmed']),axis=1)
df_all['keyword_in_titlekeys_jacnum'] = df_all['jaccard_keyword_in_titlekeys_tuple'].map(lambda x: x[0])
df_all['keyword_in_titlekeys_jaclet'] = df_all['jaccard_keyword_in_titlekeys_tuple'].map(lambda x: x[1])
df_all=df_all.drop(['jaccard_keyword_in_titlekeys_tuple'],axis=1)




def concatenate_keys(beforethekey,thekey):
    l=[beforethekey,thekey]
    if "" in l:
        l.remove("")
    if "" in l:
        l.remove("")
    return " ".join(l)

df_all['thekeys_in_title']=df_all.apply(lambda x: \
            query_in_text(concatenate_keys(x['search_term_beforethekey_stemmed'],x['search_term_thekey_stemmed']),\
                          x['product_title_stemmed']), axis=1)  
df_all['thekeys_in_description']=df_all.apply(lambda x: \
            query_in_text(concatenate_keys(x['search_term_beforethekey_stemmed'],x['search_term_thekey_stemmed']),\
                          x['product_description_stemmed']), axis=1)  
df_all['thekeys_in_bullets']=df_all.apply(lambda x: \
            query_in_text(concatenate_keys(x['search_term_beforethekey_stemmed'],x['search_term_thekey_stemmed']),\
                          x['attribute_bullets_stemmed']), axis=1)                            

df_all['thekey_in_description_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_thekey_stemmed'],x['product_description_stemmed']), axis=1) 
df_all['thekey_in_description_sum'] = df_all['thekey_in_description_tuple'].map(lambda x: x[1])
df_all['thekey_in_description_let'] = df_all['thekey_in_description_tuple'].map(lambda x: x[2])
df_all=df_all.drop(['thekey_in_description_tuple'],axis=1)

df_all['thekey_in_bullets_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_thekey_stemmed'],x['attribute_bullets_stemmed']), axis=1) 
df_all['thekey_in_bullet