# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 00:16:48 2020

@author: Emily
"""
import pandas as pd
import re
import ast

def clean_text(text):
    token = re.sub(r'[\^\\,@\‘?!\.$%_:\-“’“”]', '', text, flags=re.I)
    return token

metadata = pd.read_csv('./the-movies-dataset/movies_metadata.csv',low_memory=False)
metadata = metadata[metadata['adult'] == 'False']
metadata = metadata.drop(['adult','homepage','budget','runtime','release_date','original_language','production_countries','production_companies','spoken_languages','video','revenue','status','vote_count'],axis=1)
metadata = metadata.dropna(subset=['imdb_id','poster_path'])
metadata['tagline'] = metadata['tagline'].apply(lambda x: clean_text(str(x)).lower())
metadata['title'] = metadata['title'].apply(lambda x: clean_text(str(x)).lower())
metadata['original_title'] = metadata['original_title'].apply(lambda x: clean_text(str(x)).lower())
metadata['overview'] = metadata['overview'].apply(lambda x: clean_text(str(x)).lower())
metadata['genres'] = metadata['genres'].apply(lambda x: ast.literal_eval(x)) #make into dictionary
metadata['genres'] = metadata['genres'].apply(lambda x: ', '.join([d['name'] for d in x]))
metadata['imdbURL'] = 'https://www.imdb.com/title/' + metadata['imdb_id'] + '/'
metadata['tmdbURL'] = 'https://www.themoviedb.org/movie/' + metadata['id']
metadata['ImageURL'] = 'https://image.tmdb.org/t/p/w92' + metadata['poster_path']
#ratings = pd.read_csv('./the-movies-dataset/ratings.csv',low_memory=False)


metadata.to_csv('./the-movies-dataset/metadata_prep.csv')