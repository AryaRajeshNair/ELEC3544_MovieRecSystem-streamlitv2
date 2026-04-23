import pandas as pd
import numpy as np


def compute_model_overlap(content_indices, embedding_indices, hybrid_indices):
    """
    Compute overlap statistics between the three models.
    
    Returns:
        dict: Statistics including overlap sets and percentages
    """
    content_set = set(map(int, content_indices))
    embedding_set = set(map(int, embedding_indices))
    hybrid_set = set(map(int, hybrid_indices))
    
    # Pairwise overlaps
    content_embedding = content_set.intersection(embedding_set)
    content_hybrid = content_set.intersection(hybrid_set)
    embedding_hybrid = embedding_set.intersection(hybrid_set)
    
    # Three-way overlap
    all_three = content_set.intersection(embedding_set).intersection(hybrid_set)
    
    # Unique to each model
    content_unique = content_set - embedding_set - hybrid_set
    embedding_unique = embedding_set - content_set - hybrid_set
    hybrid_unique = hybrid_set - content_set - embedding_set
    
    # Total recommendations per model
    total_per_model = len(content_set)
    
    return {
        'content_set': content_set,
        'embedding_set': embedding_set,
        'hybrid_set': hybrid_set,
        'all_three': all_three,
        'content_embedding': content_embedding,
        'content_hybrid': content_hybrid,
        'embedding_hybrid': embedding_hybrid,
        'content_unique': content_unique,
        'embedding_unique': embedding_unique,
        'hybrid_unique': hybrid_unique,
        'agreement_percent': len(all_three) / total_per_model * 100 if total_per_model > 0 else 0
    }


def build_model_comparison_table(df, content_indices, content_scores, embedding_indices, embedding_scores, hybrid_indices, hybrid_scores):
    """Build a comprehensive comparison table of all model recommendations."""
    # Convert to numpy arrays for consistent indexing
    content_indices = np.asarray(content_indices).flatten()
    embedding_indices = np.asarray(embedding_indices).flatten()
    hybrid_indices = np.asarray(hybrid_indices).flatten()
    content_scores = np.asarray(content_scores).flatten()
    embedding_scores = np.asarray(embedding_scores).flatten()
    hybrid_scores = np.asarray(hybrid_scores).flatten()
    
    comparison_data = []
    
    for i in range(len(content_indices)):
        content_idx = int(content_indices[i])
        embedding_idx = int(embedding_indices[i])
        hybrid_idx = int(hybrid_indices[i])
        
        row = {
            'Rank': i + 1,
            'Content-Based': df.iloc[content_idx]['title'],
            'Content Score': f"{float(content_scores[i]):.1f}%",
            'Semantic Embedding': df.iloc[embedding_idx]['title'],
            'Embedding Score': f"{float(embedding_scores[i]):.1f}%",
            'Popularity Hybrid': df.iloc[hybrid_idx]['title'],
            'Hybrid Score': f"{float(hybrid_scores[i]):.1f}%"
        }
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


