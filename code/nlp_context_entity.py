# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:40:58 2020

@author: Administrator
"""

# intent mapping
import spacy
import re 
import pandas as pd
import numpy as np

# embeddings from 
df = pd.read_csv('./the-movies-dataset/df_prep.csv',low_memory=False)
nlp = spacy.load("en_core_web_md")
placeholder = [x for x in df['genres'] if isinstance(x,str)]
genres = list(set((', '.join(placeholder)).split(", ")))
titles = df['title'].tolist()
insults = ['you suck','stupid','idiot','dumb']

# get array of glove vectors for a list of strings
def vectorise(list_of_strings):
    list_of_string_vecs = np.zeros((len(list_of_strings),300))
    for i in range(len(list_of_strings)):
        list_of_string_vecs[i] = np.array(nlp(list_of_strings[i]).vector)
    return list_of_string_vecs

# pre-computed for titles
#title_vecs = vectorise(titles)
#np.save('title_vecs_short.npy',title_vecs)
title_vecs = np.load('title_vecs_short.npy')

def clean_text(text):
    token = re.sub(r'[\^\\,@\‘?!\.$%_:\-“’“”]', '', text, flags=re.I)
    return token

def calculate_similarity(text1,array):
    base = nlp(clean_text(text1).lower()).vector
    rep_base = np.tile(base,(array.shape[0],1)) # python
    rep_base_norm = np.sqrt((rep_base**2).sum(axis=1))
    array_norm = np.sqrt((array**2).sum(axis=1))
    array_norm[array_norm==0]=100000000
    c = np.multiply(rep_base,array).sum(axis=1)/(rep_base_norm*array_norm)
    return c


def last_context(session_df,user):
    try:
        a = session_df['context'][session_df['user'] == user]
        return a.iloc[-1]
    except:
        return ' '

def last_entity(session_df,user):
    try:
        a = session_df['entity'][session_df['user'] == user]
        return a.iloc[-1]
    except:
        return ' '
        
def get_context_entity(text, session_df, user): # NLP
    global insults
    global genres
    b = calculate_similarity(clean_text(text).lower(), vectorise(insults))
    if last_context(session_df,user) == 'start_conversation':
        a = calculate_similarity(clean_text(text).lower(), vectorise(genres))
        a = np.nan_to_num(a)
        if a.max()>0.5:
            context = 'information_genre'
            entity = genres[np.argmax(a)]
        elif b.max()>0.5:
            context = 'insult'
            entity = ''
        else:
            context = 'incomprehensible'
            entity = ''
    elif last_context(session_df,user) == 'information_genre':
        a = calculate_similarity(clean_text(text).lower(),title_vecs)
        a = np.nan_to_num(a)
        if a.max()>0.9:
            context = 'information_other_movie'
            entity = [titles[np.argmax(a)],[np.argmax(a)]]
        elif b.max()>0.5:
            context = 'insult'
            entity = ''
        else:
            context = 'incomprehensible'
            entity = ''
    return context,entity
