# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 00:16:48 2020

@author: Emily
"""
import pandas as pd

metadata = pd.read_csv('./the-movies-dataset/movies_metadata.csv',low_memory=False)
metadata.info()

metadata = metadata.drop(['belongs_to_collection','homepage','popularity','tagline','status'],axis=1)
metadata = metadata.drop(['runtime','release_date','original_language','production_countries','production_companies','spoken_languages','video'],axis=1)

metadata = metadata.dropna(subset=['imdb_id','poster_path'])

pd.set_option('display.max_colwidth', -1)
print(metadata['tagline'])