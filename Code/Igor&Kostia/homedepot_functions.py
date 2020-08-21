# -*- coding: utf-8 -*-
"""
Some functions are saved in this file.
Competition: HomeDepot Search Relevance
Author: Igor Buinyi
Team: Turing test
"""


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from nltk.stem.snowball import SnowballStemmer
import nltk
from time import time
import re
import os
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
stoplist.append('till')


import difflib

stemmer = SnowballStemmer('english')



#remove 'mirr edge' from Google dict

### this basic parser is used to create spell check dictionary and or to find unique brands/materials    
### !!! the output is not lowercase    
def simple_parser(s):
    s = re.sub('&amp;', '&', s)
    s = re.sub('&nbsp;', '', s)
    s = re.sub('&#39;', '', s)
    s = s.replace("-"," ")
    s = s.replace("+"," ")
    s = re.sub(r'(?<=[a-zA-Z])\/(?=[a-zA-Z])', ' ', s)
    s = re.sub(r'(?<=\))(?=[a-zA-Z0-9])', ' ', s) # add space between parentheses and letters
    s = re.sub(r'(?<=[a-zA-Z0-9])(?=\()', ' ', s) # add space between parentheses and letters
    s = re.sub(r'(?<=[a-zA-Z][\.\,])(?=[a-zA-Z])', ' ', s) # add space after dot or colon between letters
    s = re.sub('[^a-zA-Z0-9\n\ ]', '', s)
    return s

"""    
The following function creates dict for brands or material
 the entry format:
 {brand/material_name: 
        {'name': brand/material_name,
         'nwords': number of words in name,
         'cnt_attribute': number of occurencies in brand/material attributes,
         'cnt_query': number of occurencies in query,
         'cnt_text': number of occurencies in some other field like product title
        }}
"""
def get_attribute_dict(list_of_attributes,str_query,str_sometext="",search_in_text=False):
    t2 = time()
    list_of_uniq_attributes = list(set(list_of_attributes))
    list_of_uniq_attributes.remove("")
    attributes_dict={}
    cnt=0
    for attribute in list_of_uniq_attributes:
        cnt+=1
        attributes_dict[attribute]={}
        attributes_dict[attribute]['name']=attribute
        attributes_dict[attribute]['nwords']=len(attribute.split())
        attributes_dict[attribute]['cnt_attribute']=list_of_attributes.count(attribute)
        if search_in_text:
            attributes_dict[attribute]['cnt_text']=len(re.findall(r'\b'+attribute+r'\b',str_sometext))
        attributes_dict[attribute]['cnt_query']=len(re.findall(r'\b'+attribute+r'\b',str_query))
        if (cnt % 500)==0:
            print ""+str(cnt)+" out of "+str(len(list_of_uniq_attributes))+" unique attributes",round((time()-t2)/60,1) ,'minutes'
    return attributes_dict


### The following function is used inside str_parser to make spell corrections in search term.
### automatic_spell_check_dict to be generated within the code.
### This function was disclosed on the forum
def spell_correction(s, automatic_spell_check_dict={}):
   
    s=s.replace("ttt","tt")    
    s=s.replace("lll","ll") 
    s=s.replace("nnn","nn") 
    s=s.replace("rrr","rr") 
    s=s.replace("sss","ss") 
    s=s.replace("zzz","zz")
    s=s.replace("ccc","cc")
    s=s.replace("eee","ee")

    s=s.replace("hinges with pishinges with pins","hinges with pins")    
    s=s.replace("virtue usa","virtu usa")
    s = re.sub('outdoor(?=[a-rt-z])', 'outdoor ', s)
    s=re.sub(r'\bdim able\b',"dimmable", s) 
    s=re.sub(r'\blink able\b',"linkable", s)
    s=re.sub(r'\bm aple\b',"maple", s)
    s=s.replace("aire acondicionado", "air conditioner")
    s=s.replace("borsh in dishwasher", "bosch dishwasher")
    s=re.sub(r'\bapt size\b','appartment size', s)
    s=re.sub(r'\barm[e|o]r max\b','armormax', s)
    s=re.sub(r' ss ',' stainless steel ', s)
    s=re.sub(r'\bmay tag\b','maytag', s)
    s=re.sub(r'\bback blash\b','backsplash', s)
    s=re.sub(r'\bbum boo\b','bamboo', s)
    s=re.sub(r'(?<=[0-9] )but\b','btu', s)
    s=re.sub(r'\bcharbroi l\b','charbroil', s)
    s=re.sub(r'\bair cond[it]*\b','air conditioner', s)
    s=re.sub(r'\bscrew conn\b','screw connector', s)
    s=re.sub(r'\bblack decker\b','black and decker', s)
    s=re.sub(r'\bchristmas din\b','christmas dinosaur', s)
    s=re.sub(r'\bdoug fir\b','douglas fir', s)
    s=re.sub(r'\belephant ear\b','elephant ears', s)
    s=re.sub(r'\bt emp gauge\b','temperature gauge', s)
    s=re.sub(r'\bsika felx\b','sikaflex', s)
    s=re.sub(r'\bsquare d\b', 'squared', s)
    s=re.sub(r'\bbehring\b', 'behr', s)
    s=re.sub(r'\bcam\b', 'camera', s)
    s=re.sub(r'\bjuke box\b', 'jukebox', s)
    s=re.sub(r'\brust o leum\b', 'rust oleum', s)
    s=re.sub(r'\bx mas\b', 'christmas', s)
    s=re.sub(r'\bmeld wen\b', 'jeld wen', s)
    s=re.sub(r'\bg e\b', 'ge', s)
    s=re.sub(r'\bmirr edge\b', 'mirredge', s)
    s=re.sub(r'\bx ontrol\b', 'control', s)
    s=re.sub(r'\boutler s\b', 'outlets', s)
    s=re.sub(r'\bpeep hole', 'peephole', s)
    s=re.sub(r'\bwater pik\b', 'waterpik', s)
    s=re.sub(r'\bwaterpi k\b', 'waterpik', s)
    s=re.sub(r'\bplex[iy] glass\b', 'plexiglass', s)
    s=re.sub(r'\bsheet rock\b', 'sheetrock',s)
    s=re.sub(r'\bgen purp\b', 'general purpose',s)
    s=re.sub(r'\bquicker crete\b', 'quikrete',s)
    s=re.sub(r'\bref ridge\b', 'refrigerator',s)
    s=re.sub(r'\bshark bite\b', 'sharkbite',s)
    s=re.sub(r'\buni door\b', 'unidoor',s)
    s=re.sub(r'\bair tit\b','airtight', s)
    s=re.sub(r'\bde walt\b','dewalt', s)
    s=re.sub(r'\bwaterpi k\b','waterpik', s)
    s=re.sub(r'\bsaw za(ll|w)\b','sawzall', s)
    s=re.sub(r'\blg elec\b', 'lg', s)
    s = re.sub(r'\bhumming bird\b', 'hummingbird', s)
    s = re.sub(r'\bde ice(?=r|\b)', 'deice',s)  
    s = re.sub(r'\bliquid nail\b', 'liquid nails', s)  
    
    
    s=re.sub(r'\bdeck over\b','deckover', s)
    s=re.sub(r'\bcounter sink(?=s|\b)','countersink', s)
    s=re.sub(r'\bpipes line(?=s|\b)','pipeline', s)
    s=re.sub(r'\bbook case(?=s|\b)','bookcase', s)
    s=re.sub(r'\bwalkie talkie\b','2 pair radio', s)
    s=re.sub(r'(?<=^)ks\b', 'kwikset',s)
    s = re.sub('(?<=[0-9])[\ ]*ft(?=[a-z])', 'ft ', s)
    s = re.sub('(?<=[0-9])[\ ]*mm(?=[a-z])', 'mm ', s)
    s = re.sub('(?<=[0-9])[\ ]*cm(?=[a-z])', 'cm ', s)
    s = re.sub('(?<=[0-9])[\ ]*inch(es)*(?=[a-z])', 'in ', s)
    
    s = re.sub(r'(?<=[1-9]) pac\b', 'pack', s)
 
    s = re.sub(r'\bcfl bulbs\b', 'cfl light bulbs', s)
    s = re.sub(r' cfl(?=$)', ' cfl light bulb', s)
    s = re.sub(r'candelabra cfl 4 pack', 'candelabra cfl light bulb 4 pack', s)
    s = re.sub(r'\bthhn(?=$|\ [0-9]|\ [a-rtuvx-z])', 'thhn wire', s)
    s = re.sub(r'\bplay ground\b', 'playground',s)
    s = re.sub(r'\bemt\b', 'emt electrical metallic tube',s)
    s = re.sub(r'\boutdoor dining se\b', 'outdoor dining set',s)
          
    if "a/c" in s:
        if ('unit' in s) or ('frost' in s) or ('duct' in s) or ('filt' in s) or ('vent' in s) or ('clean' in s) or ('vent' in s) or ('portab' in s):
            s=s.replace("a/c","air conditioner")
        else:
            s=s.replace("a/c","ac")

   
    external_data_dict={'airvents': 'air vents', 
    'antivibration': 'anti vibration', 
    'autofeeder': 'auto feeder', 
    'backbrace': 'back brace', 
    'behroil': 'behr oil', 
    'behrwooden': 'behr wooden', 
    'brownswitch': 'brown switch', 
    'byefold': 'bifold', 
    'canapu': 'canopy', 
    'cleanerakline': 'cleaner alkaline',
    'colared': 'colored', 
    'comercialcarpet': 'commercial carpet', 
    'dcon': 'd con', 
    'doorsmoocher': 'door smoocher', 
    'dreme': 'dremel', 
    'ecobulb': 'eco bulb', 
    'fantdoors': 'fan doors', 
    'gallondrywall': 'gallon drywall', 
    'geotextile': 'geo textile', 
    'hallodoor': 'hallo door', 
    'heatgasget': 'heat gasket', 
    'ilumination': 'illumination', 
    'insol': 'insulation', 
    'instock': 'in stock', 
    'joisthangers': 'joist hangers', 
    'kalkey': 'kelkay', 
    'kohlerdrop': 'kohler drop', 
    'kti': 'kit', 
    'laminet': 'laminate', 
    'mandoors': 'main doors', 
    'mountspacesaver': 'mount space saver', 
    'reffridge': 'refrigerator', 
    'refrig': 'refrigerator', 
    'reliabilt': 'reliability', 
    'replaclacemt': 'replacement', 
    'searchgalvanized': 'search galvanized', 
    'seedeater': 'seed eater', 
    'showerstorage': 'shower storage', 
    'straitline': 'straight line', 
    'subpumps': 'sub pumps', 
    'thromastate': 'thermostat', 
    'topsealer': 'top sealer', 
    'underlay': 'underlayment',
    'vdk': 'bdk', 
    'wallprimer': 'wall primer', 
    'weedbgon': 'weed b gon', 
    'weedeaters': 'weed eaters', 
    'weedwacker': 'weed wacker', 
    'wesleyspruce': 'wesley spruce', 
    'worklite': 'work light'}
         
    for word in external_data_dict.keys():
        s=re.sub(r'\b'+word+r'\b',external_data_dict[word], s)
        
    ############ replace words from dict
    for word in automatic_spell_check_dict.keys():
        s=re.sub(r'\b'+word+r'\b',automatic_spell_check_dict[word], s)
   
    return s

"""
The following function contains some replacement to be made in all text (not only search terms).
Most of the replacements are not shared on the forum because they are thesaurus replacements, not spell correction.
"""
def replace_in_parser(s):
    #the first three shared on forum
    s=s.replace("acccessories","accessories")
    s = re.sub(r'\bscott\b', 'scotts', s) #brand
    s = re.sub(r'\borgainzer\b', 'organizer', s)
    
    # the others are not shared
    s = re.sub(r'\aluminuum\b', 'aluminum', s)    
    s = re.sub(r'\bgeneral electric','ge', s)
    s = s.replace("adaptor","adapter")
    s = re.sub(r'\bfibre', 'fiber', s)
    s = re.sub(r'\bbuilt in\b', 'builtin',s)
    s = re.sub(r'\bshark bite\b', 'sharkbite',s)
    s = re.sub('barbeque', 'barbecue',s)
    s = re.sub(r'\bbbq\b', 'barbecue', s)
    s = re.sub(r'\bbathroom[s]*\b', 'bath', s)
    s = re.sub(r'\bberkeley\b', 'berkley', s)
    s = re.sub(r'\bbookshelves\b', 'book shelf', s)
    s = re.sub(r'\bbookshelf\b', 'book shelf', s)
    s = re.sub(r'\bin line ', ' inline ', s)
    s = re.sub(r'round up\b', ' roundup', s)
    s = re.sub(r'\blg electronics\b', 'lg', s)
    s = re.sub(r'\bhdtv\b', 'hd tv', s)
    s = re.sub(r'black [and ]*decker', 'black and decker', s)
    s = re.sub(r'backer board[s]*', 'backerboard', s)
    s = re.sub(r'\bphillips\b', 'philips', s)
    s = re.sub(r'\bshower head[s]*\b', 'showerhead', s)
    s = re.sub(r'\bbull nose\b', 'bullnose', s)
    s = re.sub(r'\bflood light\b', 'floodlight', s)
    s = re.sub(r'\barrester\b', 'arrestor', s)
    s = re.sub(r'\bbi fold\b', 'bifold', s)
    s = re.sub(r'\bfirepit[s]*\b', 'fire pit', s)
    s = re.sub(r'\bbed bug[s]*\b', 'bedbug', s)
    s = re.sub(r'\bhook up[s]*\b', 'hookup', s)
    s = re.sub(r'\bjig saw[s]*\b', 'jigsaw', s)
    s = re.sub(r'\bspacesav(?=er[s]*|ing)', 'space sav', s)
    s = re.sub(r'\bwall paper', 'wallpaper', s)
    s = re.sub(r'\bphotocell', 'photo cells', s)
    s = re.sub(r'\bplasti dip\b', 'plastidip', s)
    s = re.sub(r'\bflexi dip\b', 'flexidip', s)  
    s = re.sub(r'\bback splash','backsplash', s)
    s = re.sub(r'\bbarstool(?=\b|s)','bar stool', s)
    s = re.sub(r'\blampholder(?=\b|s)','lamp holder', s)
    s = re.sub(r'\brainsuit(?=\b|s)','rain suit', s)
    s = re.sub(r'\bback up\b','backup', s)
    s = re.sub(r'\bwheel barrow', 'wheelbarrow', s)
    s=re.sub(r'\bsaw horse', 'sawhorse',s)
    s=re.sub(r'\bscrew driver', 'screwdriver',s)
    s=re.sub(r'\bnut driver', 'nutdriver',s)
    s=re.sub(r'\bflushmount', 'flush mount',s)
    s=re.sub(r'\bcooktop(?=\b|s\b)', 'cook top',s)
    s=re.sub(r'\bcounter top(?=s|\b)','countertop', s)    
    s=re.sub(r'\bbacksplash', 'back splash',s)
    s=re.sub(r'\bhandleset', 'handle set',s)
    s=re.sub(r'\bplayset', 'play set',s)
    s=re.sub(r'\bsidesplash', 'side splash',s)
    s=re.sub(r'\bdownlight', 'down light',s)
    s=re.sub(r'\bbackerboard', 'backer board',s)
    s=re.sub(r'\bshoplight', 'shop light',s)
    s=re.sub(r'\bdownspout', 'down spout',s)
    s=re.sub(r'\bpowerhead', 'power head',s)
    s=re.sub(r'\bnightstand', 'night stand',s)
    s=re.sub(r'\bmicro fiber[s]*\b', 'microfiber', s)
    s=re.sub(r'\bworklight', 'work light',s)
    s=re.sub(r'\blockset', 'lock set',s)
    s=re.sub(r'\bslatwall', 'slat wall',s)
    s=re.sub(r'\btileboard', 'tile board',s)
    s=re.sub(r'\bmoulding', 'molding',s)
    s=re.sub(r'\bdoorstop', 'door stop',s)
    s=re.sub(r'\bwork bench\b','workbench', s)
    s=re.sub(r'\bweed[\ ]*eater','weed trimmer', s)
    s=re.sub(r'\bweed[\ ]*w[h]*acker','weed trimmer', s)
    s=re.sub(r'\bnightlight(?=\b|s)','night light', s)
    s=re.sub(r'\bheadlamp(?=\b|s)','head lamp', s)
    s=re.sub(r'\bfiber board','fiberboard', s)
    s=re.sub(r'\bmail box','mailbox', s)
    
    replace_material_dict={'aluminium': 'aluminum', 
    'medium density fiberboard': 'mdf',
    'high density fiberboard': 'hdf',
    'fiber reinforced polymer': 'frp',
    'cross linked polyethylene': 'pex',
    'poly vinyl chloride': 'pvc', 
    'thermoplastic rubber': 'tpr', 
    'poly lactic acid': 'pla', 
    'acrylonitrile butadiene styrene': 'abs',
    'chlorinated poly vinyl chloride': 'cpvc'}
    for word in replace_material_dict.keys():
        if word in s:
            s = s.replace(word, replace_material_dict[word])
    
    return s


"""
The following function used to process the all text fields
"""
def str_parser(s, automatic_spell_check_dict={}, remove_from_brackets=False,parse_material=False,add_space_stop_list=[]):
    #the following three replacements are shared on the forum    
    s = s.replace("craftsm,an","craftsman")        
    s = re.sub(r'depot.com/search=', '', s)
    s = re.sub(r'pilers,needlenose', 'pliers, needle nose', s)
    
    s = re.sub(r'\bmr.', 'mr ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub('&nbsp;', '', s)
    s = re.sub('&#39;', '', s)
    s = re.sub(r'(?<=[0-9]),[\ ]*(?=[0-9])', '', s)
    s = s.replace(";",".")
    s = s.replace(",",".")
    s = s.replace(":",". ")
    s = s.replace("+"," ")
    s = re.sub(r'\bU.S.', 'US ', s)
    s = s.replace(" W x "," ")
    s = s.replace(" H x "," ")
    s = re.sub(' [\#]\d+[\-\d]*[\,]*', '', s)    
    s = re.sub('(?<=[0-9\%])(?=[A-Z][a-z])', '. ', s) # add dot between number and cap letter
    s = re.sub(r'(?<=\))(?=[a-zA-Z0-9])', ' ', s) # add space between parentheses and letters
    s = re.sub(r'(?<=[a-zA-Z0-9])(?=\()', ' ', s) # add space between parentheses and letters

    if parse_material:
        replace_dict={'Medium Density Fiberboard (MDF)':'mdf', 'High Density Fiberboard (HDF)':'hdf',\
        'Fibre Reinforced Polymer (FRP)': 'frp', 'Acrylonitrile Butadiene Styrene (ABS)': 'abs',\
        'Cross-Linked Polyethylene (PEX)':'pex', 'Chlorinated Poly Vinyl Chloride (CPVC)': 'cpvc',\
        'PVC (vinyl)': 'pvc','Thermoplastic rubber (TPR)':'tpr','Poly Lactic Acid (PLA)': 'pla',\
        '100% Polyester':'polyester','100% UV Olefin':'olefin', '100% BCF Polypropylene': 'polypropylene',\
        '100% PVC':'pvc'}
        
        if s in replace_dict.keys():
            s=replace_dict[s]


    s = re.sub('[^a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)]', ' ', s)
    s= " ".join(s.split())

    s=s.replace("-"," ")
    
    if len(add_space_stop_list)>0:
        s = " ".join([re.sub('(?<=[a-z])(?=[A-Z][a-z\ ])', '. ', word)  if word.lower() not in add_space_stop_list else word for word in s.split()])

    s=s.lower() 
    s = re.sub('\.(?=[a-z])', '. ', s) #dots before words -> replace with spaces
   # s = re.sub('(?<=[a-z])(?=[A-Z][a-z\ ])', ' ', s) # add space if uppercase after lowercase
    s = re.sub('(?<=[a-z][a-z][a-z])(?=[0-9])', ' ', s) # add cpase if number after at least three letters
    ##s = re.sub('(?<=[a-zA-Z])\.(?=\ |$)', '', s) #remove dots at the end of string
    #s = re.sub('(?<=[0-9])\.(?=\ |$)', '', s) # dot after digit before space
    s = re.sub('^\.\ ', '', s) #dot at the beginning before space
    

    if len(automatic_spell_check_dict.keys())>0:
        s=spell_correction(s,automatic_spell_check_dict=automatic_spell_check_dict)
    
    if remove_from_brackets==True:
        s = re.sub('(?<=\()[a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)]*(?=\))', '', s)
    else:
        s=s.replace(" (",". ")
        s=re.sub('(?<=[a-zA-Z0-9\%\$])\(', '. ', s)
        s=s.replace(" )",". ")
        s=s.replace(")",". ")
        s=s.replace("  "," ")
        s = re.sub('\ \.', '\.', s)
        

    #######s = re.sub('(?<=[0-9\%])(?=[a-wyz])', ' ', s) # add space between number and text (except letter x) 
    #s = re.sub('(?<=[a-zA-Z])-(?=[a-zA-Z])', ' ', s) # replace '-' in words with space
    s=s.replace("at&t","att")
    s=s.replace("&"," and ")    
    s=s.replace("*"," x ")
    s = re.sub('(?<=[a-z\ ])\/(?=[a-z\ ])', ' ', s) # replace "/" between words with space
    s = re.sub('(?<=[a-z])\\\\(?=[a-z])', ' ', s) # replace "/" between words with space
    s=s.replace("  "," ")
    s=s.replace("  "," ")
    
    #s=re.sub('(?<=\ [a-ux-z])\ (?=[0-9])', '', s)   #remove spaces
    #s=re.sub('(?<=^[a-z])\ (?=[0-9])', '', s)   #remove spaces




    #####################################
    ### thesaurus replacement in all vars
    s=replace_in_parser(s)
    
    s = re.sub('half(?=\ inch)', '1/2', s)
    s = re.sub('\ba half\b', '1/2', s)
    #s = re.sub('half\ ', 'half-', s)

    s = re.sub(r'(?<=\')s\b', '', s)
    s = re.sub('(?<=[0-9])\'\'', ' in ', s)
    s = re.sub('(?<=[0-9])\'', ' in ', s)

    s = re.sub(r'(?<=[0-9])[\ ]*inch[es]*\b', '-in ', s)
    s = re.sub(r'(?<=[0-9])[\ ]*in\b', '-in ', s)
    
    s = re.sub(r'(?<=[0-9])[\-|\ ]*feet[s]*\b', '-ft ', s)
    s = re.sub(r'(?<=[0-9])[\ ]*foot[s]*\b', '-ft ', s)
    s = re.sub(r'(?<=[0-9])[\ ]*ft[x]*\b', '-ft ', s)
    
    s = re.sub('(?<=[0-9])[\ ]*volt[s]*(?=\ |$|\.)', '-V ', s)
    s = re.sub('(?<=[0-9])[\ ]*v(?=\ |$|\.)', '-V ', s)
    
    s = re.sub('(?<=[0-9])[\ ]*wat[t]*[s]*(?=\ |$|\.)', '-W ', s)
    s = re.sub('(?<=[0-9])[\ ]*w(?=\ |$|\.)', '-W ', s)
    
    s = re.sub('(?<=[0-9])[\ ]*kilo[\ ]*watt[s]*(?=\ |$|\.)', '-KW ', s)
    s = re.sub('(?<=[0-9])[\ ]*kw(?=\ |$|\.)', '-KW ', s)
    
    s = re.sub('(?<=[0-9])[\ ]*amp[s]*(?=\ |$|\.)', '-A ', s)
    #s = re.sub('(?<=[0-9]) a(?=\ |$|\.)', '-A. ', s)
    s = re.sub('(?<=[0-9])a(?=\ |$|\.)', '-A ', s)

    s = re.sub('(?<=[0-9])[\ ]*gallon[s]*(?=\ |$|\.)', '-gal ', s)
    s = re.sub('(?<=[0-9])[\ ]*gal(?=\ |$|\.)', '-gal ', s)
        
    s = re.sub('(?<=[0-9])[\ ]*pound[s]*(?=\ |$|\.)', '-lb ', s)
    s = re.sub('(?<=[0-9])[\ ]*lb[s]*(?=\ |$|\.)', '-lb ', s)
        
    s = re.sub('(?<=[0-9])[\ ]*mi[l]+imet[er]*[s]*(?=\ |$|\.)', '-mm ', s)
    s = re.sub('(?<=[0-9])[\ ]*mm(?=\ |$|\.)', '-mm ', s)
        
    s = re.sub('(?<=[0-9])[\ ]*centimeter[s]*(?=\ |$|\.)', '-cm ', s)
    s = re.sub('(?<=[0-9])[\ ]*cm(?=\ |$|\.)', '-cm ', s)
        
    s = re.sub('(?<=[0-9])[\ ]*ounce[s]*(?=\ |$|\.)', '-oz ', s)
    s = re.sub('(?<=[0-9])[\ ]*oz(?=\ |$|\.)', '-oz ', s)
    
    s = re.sub('(?<=[0-9])[\ ]*liter[s]*(?=\ |$|\.)', '-L ', s)
    s = re.sub('(?<=[0-9])[\ ]*litre[s]*(?=\ |$|\.)', '-L ', s)
    s = re.sub('(?<=[0-9])[\ ]*l(?=\ |$|\.)', '-L. ', s)
    
    s = re.sub('(?<=[0-9])[\ ]*square feet[s]*(?=\ |$|\.)', '-sqft ', s)
    s = re.sub('(?<=[0-9])square feet[s]*(?=\ |$|\.)', '-sqft ', s)
    s = re.sub('(?<=[0-9])[\ ]*sq[\ |\.|\.\ ]*ft(?=\ |$|\.)', '-sqft ', s)
    s = re.sub('(?<=[0-9])[\ ]*sq. ft(?=\ |$|\.)', '-sqft', s)
    s = re.sub('(?<=[0-9])[\ ]*sq.ft(?=\ |$|\.)', '-sqft', s)
    
    s = re.sub('(?<=[0-9])[\ ]*cubic f[e]*t[s]*(?=\ |$|\.)', '-cuft ', s)
    s = re.sub('(?<=[0-9])[\ ]*cu[\ |\.|\.\ ]*ft(?=\ |$|\.)', '-cuft ', s)
    s = re.sub('(?<=[0-9])[\ ]*cu[\.]*[\ ]*ft(?=\ |$|\.)', '-cuft', s)
    
     
    #remove 'x'
    s = re.sub('(?<=[0-9]) x (?=[0-9])', '-X ', s)
    s = re.sub('(?<=[0-9])x (?=[0-9])', '-X ', s)
    s = re.sub('(?<=[0-9]) x(?=[0-9])', '-X ', s)
    s = re.sub('(?<=[0-9])x(?=[0-9])', '-X ', s)
    
    #s=s.replace("..",".")
    s=s.replace("\n"," ")
    s=s.replace("  "," ")

    words=s.split()

    if s.find("-X")>=0:
        for cnt in range(0,len(words)-1):
            if words[cnt].find("-X")>=0:
                if words[cnt+1].find("-X") and cnt<len(words)-2:
                    cntAdd=2
                else:
                    cntAdd=1
                to_replace=re.search(r'(?<=[0-9]\-)\w+\b',words[cnt+cntAdd])
                if not (to_replace==None):
                    words[cnt]=words[cnt].replace("-X","-"+to_replace.group(0)+"")
                else:
                    words[cnt]=words[cnt].replace("-X","x")
    s = " ".join([word for word in words])
    
    s = re.sub('[^a-zA-Z0-9\ \%\$\-\@\&\/\.]', '', s) #remove "'" and "\n" and "#" and characters
    ##s = re.sub('(?<=[a-zA-Z])[\.|\/](?=\ |$)', '', s) #remove dots at the end of string
    s = re.sub('(?<=[0-9])x(?=\ |$)', '', s) #remove 
    s = re.sub('(?<=[\ ])x(?=[0-9])', '', s) #remove
    s = re.sub('(?<=^)x(?=[0-9])', '', s)
    #s = re.sub('[\ ]\.(?=\ |$)', '', s) #remove dots 
    s=s.replace("  "," ")
    s=s.replace("..",".")
    s = re.sub('\ \.', '', s)
    
    s=re.sub('(?<=\ [ch-hj-np-su-z][a-z])\ (?=[0-9])', '', s) #remove spaces
    s=re.sub('(?<=^[ch-hj-np-su-z][a-z])\ (?=[0-9])', '', s) #remove spaces
    
    s = re.sub('(?<=\ )\.(?=[0-9])', '0.', s)
    s = re.sub('(?<=^)\.(?=[0-9])', '0.', s)
    return " ".join([word for word in s.split()])



def cut_punctuation(s):
    return " ".join([re.sub(r'[;,.](?=$)','', word) for word in s.split()])

### str_stemmer() = stemmer(str_parser())
def str_stemmer(s, automatic_spell_check_dict={},remove_from_brackets=False,parse_material=False,add_space_stop_list=[], stoplist=stoplist):
    s=str_parser(s,automatic_spell_check_dict=automatic_spell_check_dict, remove_from_brackets=remove_from_brackets,\
            parse_material=parse_material, add_space_stop_list=add_space_stop_list)
    s=" ".join([word for word in s.split() if word not in stoplist])
    return " ".join([stemmer.stem(re.sub('\.(?=$)', '', word)) for word in s.split()])

def str_stemmer_wo_parser(s, stoplist=stoplist):
    s=" ".join([word for word in s.split() if word not in stoplist])
    return " ".join([stemmer.stem(re.sub('\.(?=$)', '', word)) for word in s.split()])


"""
There are many non-unique queries and products. To save time, in some cases we processed only unique entries.
The following function applies str_parser() function to unique entries only.
"""
def col_parser(clmn, automatic_spell_check_dict={}, remove_from_brackets=False,parse_material=False,add_space_stop_list=[]):
    t0 = time()
    aa=list(set(list(clmn)))
    my_dict={}
    for i in range(0,len(aa)):
        my_dict[aa[i]]=str_parser(aa[i],automatic_spell_check_dict=automatic_spell_check_dict, remove_from_brackets=remove_from_brackets,\
                                    parse_material=parse_material,add_space_stop_list=add_space_stop_list)
        if (i % 10000)==0:
            print "parsed "+str(i)+" out of "+str(len(aa))+" unique values; "+str(round((time()-t0)/60,1))+" minutes"
    return clmn.map(lambda x: my_dict[x])
    


"""
Function to find similarities for a pair of words.
First, for each word the lists of the corresponding WordNet synsets are retrieved.
Second, max and mean similarities for the synset pairs in the lists are calculated.
Similarity measures: path similarity, Leacock-Chodorow similarity and Resnik similarity.
"""
def find_similarity(w1,w2, nouns=True,CUT_VALUE=14.50):
    if nouns==True:
        lst1=wn.synsets(w1,pos=wn.NOUN)
        lst2=wn.synsets(w2,pos=wn.NOUN)                          
    else:
        lst1=wn.synsets(w1)
        lst2=wn.synsets(w2)       
    if w1 in ['chipper', 'chippers']:
        lst1=wn.synsets("shredder",pos=wn.NOUN)
    if w2 in ['chipper', 'chippers']:
        lst2=wn.synsets("shredder",pos=wn.NOUN)                   
        
    if w1 in ['lockset', 'locksets']:
        lst1=wn.synsets("knob",pos=wn.NOUN)
    if w2 in ['lockset', 'locksets']:
        lst2=wn.synsets("knob",pos=wn.NOUN)  
    similarities_list=[item1.path_similarity(item2)  for item1 in lst1 \
                                for  item2 in lst2 \
                                if item1.path_similarity(item2)!=None]                     

    if len(similarities_list)==0:
        max_similarity=0
        mean_similarity=0
    else:
        max_similarity=max(similarities_list)
        mean_similarity=np.mean(similarities_list)
        
    lch_similarities_list=[item1.lch_similarity(item2)  for item1 in lst1 \
                                for  item2 in lst2 \
                                if item1.pos()==item2.pos() and item1.lch_similarity(item2)!=None]                     

    if len(lch_similarities_list)==0:
        max_lch_similarity=0
        mean_lch_similarity=0
    else:
        max_lch_similarity=max(lch_similarities_list)
        mean_lch_similarity=np.mean(lch_similarities_list)
        
    res_similarities_list=[min(CUT_VALUE,item1.res_similarity(item