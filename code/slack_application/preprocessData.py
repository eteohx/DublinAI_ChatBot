# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 00:16:48 2020

@author: Emily
"""
import pandas as pd
import re
import ast
import pickle

def clean_text(text):
    token = re.sub(r'[\^\\,@\‘?!\.$%_:\-“’“”]', '', text, flags=re.I)
    return token

df = pd.read_csv('./the-movies-dataset/movies_metadata.csv',low_memory=False)
mappings = pd.read_csv('./the-movies-dataset/links.csv',low_memory=False)
mappings = mappings.drop_duplicates(subset=['tmdbId'],keep='first')
mappings.set_index('tmdbId',inplace=True)
to_movieId = mappings.to_dict()['movieId']


# embeddings from 
with open('G:/My Drive/DublinAI/Mini Projects/chatbot/nn/embeddings_smaller', 'rb') as file:
    embed, movie_to_index = pickle.load(file)
    
df = df[df['adult'] == 'FALSE']
df = df.drop(['adult','homepage','budget','runtime','release_date','original_language','production_countries','production_companies','spoken_languages','video','revenue','status','vote_count'],axis=1)
df = df.dropna(subset=['imdb_id','poster_path'])
df['tagline'] = df['tagline'].apply(lambda x: clean_text(str(x)).lower())
df['title'] = df['title'].apply(lambda x: clean_text(str(x)).lower())
df['original_title'] = df['original_title'].apply(lambda x: clean_text(str(x)).lower())
df['overview'] = df['overview'].apply(lambda x: clean_text(str(x)).lower())
df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x)) #make into dictionary
df['genres'] = df['genres'].apply(lambda x: ', '.join([d['name'] for d in x]))
df['imdbURL'] = 'https://www.imdb.com/title/' + df['imdb_id'] + '/'
df['tmdbURL'] = 'https://www.themoviedb.org/movie/' + df['id']
df['ImageURL'] = 'https://image.tmdb.org/t/p/w92' + df['poster_path']
#ratings = pd.read_csv('./the-movies-dataset/ratings.csv',low_memory=False)
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].fillna('') 
#overview_vecs = vectorise(overview)
df = df.astype({'id':'int64'})
df['movieId'] = df['id'].map(to_movieId)
df['newId'] = df['movieId'].map(movie_to_index)
df = df.dropna(subset=['newId'])
df = df.reset_index()
df = df.drop(['index'],axis=1)

a = df['belongs_to_collection'][df['belongs_to_collection'].notnull()]
indices = list(a.index)
for i in indices:
    b = str(a[i]).split(", 'poster_path'")
    df['belongs_to_collection'][i]=ast.literal_eval(b[0]+'}').get('id')

df.to_csv('./the-movies-dataset/df_prep.csv')
