import plotly.graph_objects as go


def build_model_overlap_venn(overlap_stats):
    """Build a 3-set Venn-style overlap chart using Plotly shapes."""
    content_set = overlap_stats['content_set']
    embedding_set = overlap_stats['embedding_set']
    hybrid_set = overlap_stats['hybrid_set']

    content_only = len(content_set - embedding_set - hybrid_set)
    embedding_only = len(embedding_set - content_set - hybrid_set)
    hybrid_only = len(hybrid_set - content_set - embedding_set)

    content_embedding_only = len((content_set & embedding_set) - hybrid_set)
    content_hybrid_only = len((content_set & hybrid_set) - embedding_set)
    embedding_hybrid_only = len((embedding_set & hybrid_set) - content_set)
    all_three = len(content_set & embedding_set & hybrid_set)

    fig = go.Figure()

    # Three semi-transparent circles.
    fig.add_shape(type='circle', x0=0.10, y0=0.20, x1=0.60, y1=0.80,
                  fillcolor='rgba(66, 133, 244, 0.40)', line=dict(color='rgba(66, 133, 244, 1)', width=2))
    fig.add_shape(type='circle', x0=0.40, y0=0.20, x1=0.90, y1=0.80,
                  fillcolor='rgba(52, 168, 83, 0.40)', line=dict(color='rgba(52, 168, 83, 1)', width=2))
    fig.add_shape(type='circle', x0=0.25, y0=0.45, x1=0.75, y1=1.05,
                  fillcolor='rgba(251, 188, 5, 0.40)', line=dict(color='rgba(251, 188, 5, 1)', width=2))

    fig.add_annotation(x=0.16, y=0.50, text=f"{content_only}", showarrow=False, font=dict(size=20))
    fig.add_annotation(x=0.84, y=0.50, text=f"{embedding_only}", showarrow=False, font=dict(size=20))
    fig.add_annotation(x=0.50, y=0.94, text=f"{hybrid_only}", showarrow=False, font=dict(size=20))

    fig.add_annotation(x=0.50, y=0.50, text=f"{content_embedding_only}", showarrow=False, font=dict(size=18))
    fig.add_annotation(x=0.38, y=0.67, text=f"{content_hybrid_only}", showarrow=False, font=dict(size=18))
    fig.add_annotation(x=0.62, y=0.67, text=f"{embedding_hybrid_only}", showarrow=False, font=dict(size=18))
    fig.add_annotation(x=0.50, y=0.62, text=f"{all_three}", showarrow=False, font=dict(size=20, color='black'))

    fig.add_annotation(x=0.16, y=0.10, text='Content-Based', showarrow=False, font=dict(size=13))
    fig.add_annotation(x=0.84, y=0.10, text='Semantic Embedding', showarrow=False, font=dict(size=13))
    fig.add_annotation(x=0.50, y=1.10, text='Popularity Hybrid', showarrow=False, font=dict(size=13))

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1.15], scaleanchor='x', scaleratio=1)
    fig.update_layout(
        title='Recommendation Overlap (Venn Diagram)',
        showlegend=False,
        width=760,
        height=620,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig
