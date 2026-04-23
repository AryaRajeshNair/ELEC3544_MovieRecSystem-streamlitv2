import pandas as pd
import numpy as np
import plotly.graph_objects as go


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


def build_overlap_venn_data(overlap_stats):
    """Prepare data for Venn diagram visualization."""
    return {
        'All Three': len(overlap_stats['all_three']),
        'Content & Embedding': len(overlap_stats['content_embedding'] - overlap_stats['all_three']),
        'Content & Hybrid': len(overlap_stats['content_hybrid'] - overlap_stats['all_three']),
        'Embedding & Hybrid': len(overlap_stats['embedding_hybrid'] - overlap_stats['all_three']),
        'Content Only': len(overlap_stats['content_unique']),
        'Embedding Only': len(overlap_stats['embedding_unique']),
        'Hybrid Only': len(overlap_stats['hybrid_unique'])
    }


def build_overlap_bar_chart(overlap_stats):
    """Create a bar chart showing model agreement."""
    categories = [
        'All 3 Models',
        'Content & Embedding',
        'Content & Hybrid',
        'Embedding & Hybrid',
        'Content Only',
        'Embedding Only',
        'Hybrid Only'
    ]
    
    values = [
        len(overlap_stats['all_three']),
        len(overlap_stats['content_embedding'] - overlap_stats['all_three']),
        len(overlap_stats['content_hybrid'] - overlap_stats['all_three']),
        len(overlap_stats['embedding_hybrid'] - overlap_stats['all_three']),
        len(overlap_stats['content_unique']),
        len(overlap_stats['embedding_unique']),
        len(overlap_stats['hybrid_unique'])
    ]
    
    colors = [
        '#7C3AED',  # All three - purple
        '#3B82F6',  # Content & Embedding - blue
        '#06B6D4',  # Content & Hybrid - cyan
        '#10B981',  # Embedding & Hybrid - green
        '#F59E0B',  # Content only - amber
        '#8B5CF6',  # Embedding only - violet
        '#EC4899'   # Hybrid only - pink
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Movies: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Model Agreement Analysis',
        xaxis_title='Recommendation Categories',
        yaxis_title='Number of Movies',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig


def get_model_strengths(df, content_indices, embedding_indices, hybrid_indices, overlap_stats):
    """Analyze strengths of each model based on unique recommendations."""
    strengths = {
        'Content-Based': {
            'unique_count': len(overlap_stats['content_unique']),
            'agreement': len(overlap_stats['all_three']),
            'unique_movies': [df.iloc[int(idx)]['title'] for idx in list(overlap_stats['content_unique'])[:3]]
        },
        'Semantic Embedding': {
            'unique_count': len(overlap_stats['embedding_unique']),
            'agreement': len(overlap_stats['all_three']),
            'unique_movies': [df.iloc[int(idx)]['title'] for idx in list(overlap_stats['embedding_unique'])[:3]]
        },
        'Popularity Hybrid': {
            'unique_count': len(overlap_stats['hybrid_unique']),
            'agreement': len(overlap_stats['all_three']),
            'unique_movies': [df.iloc[int(idx)]['title'] for idx in list(overlap_stats['hybrid_unique'])[:3]]
        }
    }
    
    return strengths
