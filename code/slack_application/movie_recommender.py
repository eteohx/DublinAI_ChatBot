# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:27:12 2020

@author: Administrator
"""
import numpy as np
import pandas as pd
import pickle 

def inverse_mapping(f):
    return f.__class__(map(reversed, f.items()))

# embeddings for movies
with open('G:/My Drive/DublinAI/Mini Projects/chatbot/nn/embeddings_smaller', 'rb') as file:
    embed, movie_to_index = pickle.load(file)
index_to_movie = inverse_mapping(movie_to_index)
overview_vecs = np.load('overview_vecs_short.npy')

# df with movie information
df = pd.read_csv('./the-movies-dataset/df_prep.csv',low_memory=False)
set_a = set([i for i in range(len(embed))])
set_b = set(df['newId'])
missing = list(set_a.difference(set_b)) #list of movies for which we don't have metadata
df['genres'] = df['genres'].fillna('') 
overview = df['overview'].tolist()

# calculate similarity between vector(s) at location(s) vecs_id and all other vectors, 
# excluding those in exclude
def calc_multisimilarity(vecs_id,vector,exclude=[]):
    set_a = set([i for i in range(len(vector))])
    set_b = set(list(vecs_id)+list(exclude))
    to_check = list(set_a.difference(set_b))
    list_of_dist_vecs = np.zeros((len(vecs_id),len(to_check)))
    count = 0
    for i in vecs_id:
        base = vector[int(i)]
        array = vector[to_check]
        rep_base = np.tile(base,(array.shape[0],1)) # python
        rep_base_norm = np.sqrt((rep_base**2).sum(axis=1))
        array_norm = np.sqrt((array**2).sum(axis=1))
        list_of_dist_vecs[count] = np.multiply(rep_base,array).sum(axis=1)/(rep_base_norm*array_norm)        
        count +=1
    return to_check, list_of_dist_vecs.mean(axis=0)

# main recommender function... 
def movies_similar_to(entity,k,genre='all',method = 'collab',exclude_collection = True):
    temp_df = df
    idx = entity[1]
    
    if method == 'content':        
        exclude = temp_df['vote_average'].apply(lambda x: x<5.5)
        if genre=='all':
            exclude = exclude
        else:
            exclude = (temp_df['genres'].apply(lambda x: genre not in x)) | exclude
        if exclude_collection:
            for i in idx:
                if not np.isnan(temp_df['belongs_to_collection'][i]):
                    exclude= (temp_df['belongs_to_collection']==temp_df['belongs_to_collection'][i]) | exclude
        exclude[idx] = True
        dont_check_these = list(temp_df.index[exclude])
        checked,sim_scores = calc_multisimilarity(idx,overview_vecs,dont_check_these)
        sim_scores = np.nan_to_num(sim_scores)
        sorted_movies = list(np.argsort(-sim_scores, kind='stable'))
        idxes = [checked[i] for i in sorted_movies[:k]]

    elif method == 'collab':
        vecs_id = list(map(int,list(temp_df['newId'][idx])))
        if genre=='all':
            exclude = temp_df['genres'] != temp_df['genres']
        else:
            exclude = temp_df['genres'].apply(lambda x: genre not in x)
        if exclude_collection:
            for i in idx:
                if not np.isnan(temp_df['belongs_to_collection'][i]):
                    exclude= (temp_df['belongs_to_collection']==temp_df['belongs_to_collection'][i]) | exclude
        exclude[idx] = True
        exclusions = list(map(int, list(temp_df.newId[exclude])))+missing
        checked,sim_scores = calc_multisimilarity(vecs_id,embed,exclusions)
        sorted_vecs = list(np.argsort(-sim_scores, kind='stable'))
        movie_ids = [index_to_movie[checked[i]] for i in sorted_vecs[:k]]
        idxes = np.asarray(temp_df.index[temp_df['movieId'].isin(movie_ids)])


    movies = ', '.join([''.join(x) for x in (temp_df['title'].iloc[idxes])])
    return movies,idxes



