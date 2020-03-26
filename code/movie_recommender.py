# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:27:12 2020

@author: Administrator
"""
import numpy as np
import pandas as pd
from nlp_context_entity import calculate_similarity 

metadata = pd.read_csv('./the-movies-dataset/metadata_prep.csv',low_memory=False)
metadata['overview'] = metadata['overview'].fillna('')
overview = [''.join(x) for x in metadata['overview']]
metadata['genres'] = metadata['genres'].fillna('') 
#overview_vecs = vectorise(overview)
overview_vecs = np.load('overview_vecs.npy')

def movies_similar_to(genre,entity):
    idx = entity[1]
    to_keep_a = metadata['vote_average'].apply(lambda x: x>6)
    to_keep_b = metadata['genres'].apply(lambda x: genre in x)
    to_keep = to_keep_a & to_keep_b
    to_keep[idx] = False
    
    res = [i for i, val in enumerate(to_keep) if val] 
    sim_scores = calculate_similarity(overview[idx],overview_vecs[res])
    sim_scores = np.nan_to_num(sim_scores)
    sorted_movies = list(np.argsort(-sim_scores, kind='stable'))
    res = np.array(res)
    movie_indices = res[sorted_movies[:3]]
    movies = ', '.join([''.join(x) for x in (metadata['title'].iloc[movie_indices])])
    return movies

