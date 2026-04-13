import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from models.cosine_model import CosineSimilarityModel
from models.knn_model import KNNModel


@dataclass
class EvalResult:
    model: str
    hit_rate_at_k: float
    ndcg_at_k: float
    mrr_at_k: float
    coverage_at_k: float
    intra_list_diversity: float
    novelty: float
    evaluated_profiles: int


def _split_csv_text(value):
    if not isinstance(value, str):
        return set()
    return {v.strip().lower() for v in value.split(',') if v.strip()}


def build_profiles(df: pd.DataFrame, n_profiles: int, profile_size: int, seed: int = 42):
    """Build pseudo user profiles with genre coherence for offline evaluation."""
    rng = np.random.default_rng(seed)
    genre_sets = [_split_csv_text(g) for g in df['genres']]
    all_indices = np.arange(len(df))

    profiles = []
    attempts = 0
    max_attempts = n_profiles * 30

    while len(profiles) < n_profiles and attempts < max_attempts:
        attempts += 1
        anchor = int(rng.choice(all_indices))
        anchor_genres = genre_sets[anchor]
        if not anchor_genres:
            continue

        candidates = [
            idx for idx in all_indices
            if idx != anchor and len(anchor_genres.intersection(genre_sets[int(idx)])) > 0
        ]

        if len(candidates) < profile_size - 1:
            continue

        sampled = rng.choice(candidates, size=profile_size - 1, replace=False)
        profile = [anchor] + [int(x) for x in sampled]
        profiles.append(profile)

    return profiles


def ndcg_from_rank(rank):
    if rank is None:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def mrr_from_rank(rank):
    if rank is None:
        return 0.0
    return 1.0 / rank


def compute_intra_list_diversity(feature_matrix, rec_indices):
    if len(rec_indices) < 2:
        return 0.0
    rec_features = feature_matrix[rec_indices]
    sim = cosine_similarity(rec_features)
    n = sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    mean_similarity = sim[mask].mean() if mask.any() else 0.0
    return float(1.0 - mean_similarity)


def cosine_scores_for_candidates(feature_matrix, liked_indices, candidate_indices):
    user_vector = np.mean(feature_matrix[liked_indices], axis=0).reshape(1, -1)
    candidate_vectors = feature_matrix[candidate_indices]
    return cosine_similarity(user_vector, candidate_vectors)[0]


def knn_scores_for_candidates(feature_matrix, liked_indices, candidate_indices):
    liked_set = set(int(i) for i in liked_indices)
    user_vector = np.mean(feature_matrix[liked_indices], axis=0).reshape(1, -1)

    sim_to_profile = cosine_similarity(user_vector, feature_matrix[candidate_indices])[0]

    sim_to_liked = cosine_similarity(feature_matrix[candidate_indices], feature_matrix[liked_indices])
    avg_to_liked = sim_to_liked.mean(axis=1)
    max_to_liked = sim_to_liked.max(axis=1)

    # Mild "neighbor vote" proxy: count liked items above a similarity threshold
    vote_count = (sim_to_liked >= 0.2).sum(axis=1)
    vote_factor = np.log1p(vote_count)

    raw = (0.45 * sim_to_profile) + (0.35 * avg_to_liked) + (0.20 * max_to_liked)
    scores = raw * np.maximum(vote_factor, 1.0)

    # Safety: liked items should never be candidates, but guard anyway.
    for pos, idx in enumerate(candidate_indices):
        if int(idx) in liked_set:
            scores[pos] = -np.inf

    return scores


def rank_heldout_against_negatives(model_name, feature_matrix, liked_indices, holdout_idx, negative_indices):
    candidate_indices = np.array([holdout_idx] + [int(i) for i in negative_indices], dtype=int)

    if model_name == "Cosine Similarity":
        scores = cosine_scores_for_candidates(feature_matrix, liked_indices, candidate_indices)
    else:
        scores = knn_scores_for_candidates(feature_matrix, liked_indices, candidate_indices)

    order = np.argsort(scores)[::-1]
    ranked_candidates = candidate_indices[order]
    ranked_list = ranked_candidates.tolist()
    return ranked_list.index(int(holdout_idx)) + 1


def evaluate_model(model_name, model, features_df, df, profiles, k, seed=42):
    rng = np.random.default_rng(seed)
    feature_matrix = features_df.values
    movie_count = len(df)

    pop = pd.to_numeric(df.get('vote_count', pd.Series(np.ones(movie_count))), errors='coerce').fillna(0).to_numpy()
    pop = np.clip(pop, a_min=0, a_max=None)
    pop_prob = (pop + 1.0) / np.sum(pop + 1.0)

    hits = []
    ndcgs = []
    mrrs = []
    diversities = []
    recommended_items = set()
    novelty_scores = []
    all_indices = np.arange(movie_count)

    for profile in profiles:
        holdout = int(rng.choice(profile))
        liked = [idx for idx in profile if idx != holdout]

        excluded = set(profile)
        negative_pool = [idx for idx in all_indices if int(idx) not in excluded]
        n_negatives = min(199, len(negative_pool))
        sampled_negatives = rng.choice(negative_pool, size=n_negatives, replace=False)

        rank = rank_heldout_against_negatives(
            model_name=model_name,
            feature_matrix=feature_matrix,
            liked_indices=liked,
            holdout_idx=holdout,
            negative_indices=sampled_negatives,
        )

        rec_indices, _ = model.recommend(liked, n_recommendations=k)
        rec_indices = [int(i) for i in rec_indices]

        for idx in rec_indices:
            recommended_items.add(idx)

        hits.append(1.0 if rank <= k else 0.0)
        ndcgs.append(ndcg_from_rank(rank) if rank <= k else 0.0)
        mrrs.append(mrr_from_rank(rank) if rank <= k else 0.0)
        diversities.append(compute_intra_list_diversity(feature_matrix, rec_indices))

        if rec_indices:
            novelty = float(np.mean([-np.log2(pop_prob[idx]) for idx in rec_indices]))
            novelty_scores.append(novelty)

    return EvalResult(
        model=model_name,
        hit_rate_at_k=float(np.mean(hits)) if hits else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        mrr_at_k=float(np.mean(mrrs)) if mrrs else 0.0,
        coverage_at_k=float(len(recommended_items) / movie_count) if movie_count else 0.0,
        intra_list_diversity=float(np.mean(diversities)) if diversities else 0.0,
        novelty=float(np.mean(novelty_scores)) if novelty_scores else 0.0,
        evaluated_profiles=len(profiles),
    )


def run_evaluation(data_path, feature_path, k, n_profiles, profile_size, seed):
    df = pd.read_csv(data_path)
    features_df = pd.read_csv(feature_path)

    profiles = build_profiles(df, n_profiles=n_profiles, profile_size=profile_size, seed=seed)
    if not profiles:
        raise ValueError("No valid pseudo-user profiles could be generated. Try lowering profile_size.")

    cosine_model = CosineSimilarityModel(features_df)
    knn_model = KNNModel(features_df)

    cosine_result = evaluate_model("Cosine Similarity", cosine_model, features_df, df, profiles, k, seed)
    knn_result = evaluate_model("KNN", knn_model, features_df, df, profiles, k, seed)

    summary = pd.DataFrame([
        cosine_result.__dict__,
        knn_result.__dict__,
    ])

    print("\n=== Offline Recommendation Evaluation ===")
    print(f"Profiles evaluated: {len(profiles)}")
    print(f"Top-K: {k}\n")
    print(summary.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate recommendation models with pseudo-user leave-one-out.")
    parser.add_argument("--data", default="tmdb_movies_filtered_500.csv", help="Path to movie metadata CSV")
    parser.add_argument("--features", default="movie_feature_vectors.csv", help="Path to feature vectors CSV")
    parser.add_argument("--k", type=int, default=10, help="K for top-K ranking metrics")
    parser.add_argument("--profiles", type=int, default=200, help="Number of pseudo-user profiles")
    parser.add_argument("--profile-size", type=int, default=5, help="Movies per pseudo profile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        data_path=args.data,
        feature_path=args.features,
        k=args.k,
        n_profiles=args.profiles,
        profile_size=args.profile_size,
        seed=args.seed,
    )
