import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

df = pd.read_csv("tmdb_movies_filtered_500.csv")
print(df.info())
print(df.head())

def process_genres(genre):
  if isinstance(genre, str):
    return genre.split(', ')
  return []

def process_keywords(keywords):
  if isinstance(keywords, str):
      return keywords.split(', ')
  return []

def keywords_to_text(keyword_list):
  if isinstance(keyword_list, list):
      # Remove spaces within multi-word phrases by replacing with underscores
      # "virtual reality" → "virtual_reality"
      cleaned = [k.strip().lower().replace(' ', '_') for k in keyword_list]
      return ' '.join(cleaned)
  return ''

df['genres_list'] = df['genres'].apply(process_genres)
                                      
# Multi-hot encoding
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres_list'])
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# Clean text: lowercase, remove special chars, join into a single string
df['keywords_list'] = df['keywords'].apply(process_keywords) 
df['keywords_text'] = df['keywords_list'].apply(keywords_to_text)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=500)
keywords_tfidf = tfidf.fit_transform(df['keywords_text'])
keywords_df = pd.DataFrame(keywords_tfidf.toarray(), columns=tfidf.get_feature_names_out())

# Normalizing numerical features using min-max normalization
numerical_columns = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity']
scaler = MinMaxScaler()
df_num_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_columns]), columns=numerical_columns)

# Reset indices to ensure alignment
genres_df.reset_index(drop=True, inplace=True)
keywords_df.reset_index(drop=True, inplace=True)
df_num_scaled.reset_index(drop=True, inplace=True)

# Combine
movie_features = pd.concat([genres_df, keywords_df, df_num_scaled], axis=1)

# Initial weighting experimentation
genre_weight = 1.5
keyword_weight = 2.0
rating_weight = 0.5

weighted_features = pd.concat([
    genres_df * genre_weight,
    keywords_df * keyword_weight,
    df_num_scaled * rating_weight
], axis=1)

movie_features.to_csv("movie_feature_vectors.csv", index=False)
