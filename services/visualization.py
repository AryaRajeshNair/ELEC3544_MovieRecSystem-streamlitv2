import pandas as pd
from sklearn.decomposition import PCA


def get_pca_projection(movie_features_df):
    """Project feature vectors into 2D using PCA."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(movie_features_df.values)
    return pd.DataFrame(coords, columns=['x', 'y'])
