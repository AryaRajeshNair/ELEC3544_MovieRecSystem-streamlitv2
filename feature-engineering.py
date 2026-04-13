import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("tmdb_movies_filtered_500.csv")
print(df.info())
print(df.head())

def process_genres(genre):
    if not isinstance(genre, str):
        return []
    return [g.strip() for g in genre.split(',') if g.strip()]

def process_keywords(keywords):
    if not isinstance(keywords, str):
        return []
    return [k.strip() for k in keywords.split(',') if k.strip()]

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

# Numerical feature engineering
numerical_columns = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity']
for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Log-scale highly skewed quantities before normalization
skewed_columns = ['vote_count', 'revenue', 'budget', 'popularity']
for col in skewed_columns:
        df[col] = np.log1p(df[col].clip(lower=0))

scaler = MinMaxScaler()
df_num_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_columns]), columns=numerical_columns)

# Reset indices to ensure alignment
genres_df.reset_index(drop=True, inplace=True)
keywords_df.reset_index(drop=True, inplace=True)
df_num_scaled.reset_index(drop=True, inplace=True)

# Combine
movie_features = pd.concat([genres_df, keywords_df, df_num_scaled], axis=1)

# Feature weighting (used in final exported vectors)
genre_weight = 1.5
keyword_weight = 2.0
numeric_weight = 0.5

weighted_features = pd.concat([
    genres_df * genre_weight,
    keywords_df * keyword_weight,
        df_num_scaled * numeric_weight
], axis=1)

weighted_features.to_csv("movie_feature_vectors.csv", index=False)
print(f"Saved weighted feature vectors with shape: {weighted_features.shape}")
