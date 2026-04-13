import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


class KNNModel:
    """K-Nearest Neighbors based recommendation model."""

    def __init__(self, features_df):
        self.features = features_df.values
        self.knn = NearestNeighbors(n_neighbors=6, metric='cosine')
        self.knn.fit(self.features)

    def recommend(self, liked_indices, n_recommendations=5):
        """Generate recommendations using neighbor voting + user-relative similarity scoring."""
        liked_indices = np.array(liked_indices, dtype=int)
        if liked_indices.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        n_recommendations = max(1, int(n_recommendations))
        n_samples = self.features.shape[0]
        n_neighbors = min(max(n_recommendations + 5, 10), n_samples)

        # Candidate gathering: collect neighbors for each liked movie
        candidate_votes = {}
        for idx in liked_indices:
            distances, indices = self.knn.kneighbors([self.features[idx]], n_neighbors=n_neighbors)
            for dist, cand_idx in zip(distances[0], indices[0]):
                cand_idx = int(cand_idx)
                if cand_idx in liked_indices:
                    continue
                sim = 1.0 - float(dist)
                if cand_idx not in candidate_votes:
                    candidate_votes[cand_idx] = []
                candidate_votes[cand_idx].append(sim)

        if not candidate_votes:
            return np.array([], dtype=int), np.array([], dtype=float)

        # Score by similarity to user profile + neighbor vote support
        user_vector = np.mean(self.features[liked_indices], axis=0).reshape(1, -1)
        candidate_indices = np.array(list(candidate_votes.keys()), dtype=int)
        candidate_vectors = self.features[candidate_indices]
        profile_sims = cosine_similarity(user_vector, candidate_vectors)[0]

        scores = {}
        for cand_idx, profile_sim in zip(candidate_indices, profile_sims):
            vote_sims = candidate_votes[int(cand_idx)]
            vote_strength = float(np.mean(vote_sims))
            vote_count_factor = float(np.log1p(len(vote_sims)))
            scores[int(cand_idx)] = profile_sim * vote_strength * vote_count_factor

        sorted_recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = sorted_recommendations[:n_recommendations]

        top_indices = np.array([idx for idx, _ in top], dtype=int)
        scores_array = np.array([score for _, score in top], dtype=float)
        return top_indices, scores_array
