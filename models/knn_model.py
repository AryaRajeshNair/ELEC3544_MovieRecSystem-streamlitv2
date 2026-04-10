import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNModel:
    """K-Nearest Neighbors based recommendation model."""

    def __init__(self, features_df):
        self.features = features_df.values
        self.knn = NearestNeighbors(n_neighbors=6, metric='cosine')
        self.knn.fit(self.features)

    def recommend(self, liked_indices, n_recommendations=5):
        """Generate recommendations using nearest-neighbor aggregation."""
        all_neighbors = []

        for idx in liked_indices:
            _, indices = self.knn.kneighbors([self.features[idx]], n_neighbors=n_recommendations + 2)
            all_neighbors.extend(indices[0])

        recommendations = [x for x in set(all_neighbors) if x not in liked_indices]

        scores = {}
        for idx in recommendations:
            distances, _ = self.knn.kneighbors([self.features[idx]], n_neighbors=2)
            scores[idx] = np.mean(1 - distances[0])

        sorted_recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = np.array([x[0] for x in sorted_recommendations[:n_recommendations]])
        scores_array = np.array([x[1] for x in sorted_recommendations[:n_recommendations]])

        return top_indices, scores_array
