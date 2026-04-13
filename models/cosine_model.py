import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarityModel:
    """Cosine Similarity based recommendation model."""

    def __init__(self, features_df):
        self.features = features_df.values

    def recommend(self, liked_indices, n_recommendations=5):
        """Generate recommendations using cosine similarity to user profile vector."""
        liked_indices = np.array(liked_indices, dtype=int)
        if liked_indices.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        n_recommendations = max(1, int(n_recommendations))

        liked_features = self.features[liked_indices]
        user_vector = np.mean(liked_features, axis=0).reshape(1, -1)

        similarities = cosine_similarity(user_vector, self.features)[0]
        similarities[liked_indices] = -1

        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        return top_indices, similarities[top_indices]
