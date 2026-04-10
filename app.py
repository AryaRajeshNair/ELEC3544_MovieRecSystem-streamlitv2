import streamlit as st
import pandas as pd
import plotly.express as px

from models.cosine_model import CosineSimilarityModel
from models.knn_model import KNNModel
from services.explanations import build_recommendation_explanations
from services.recommendation_utils import (
    analyze_user_taste,
    format_recommendations,
    get_movie_suggestions,
)
from services.visualization import (
    build_overlap_figure,
    build_visualization_figure,
    get_pca_projection,
)

# Set page config
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    """Load movie data and feature vectors"""
    df = pd.read_csv("tmdb_movies_filtered_500.csv")
    movie_features = pd.read_csv("movie_feature_vectors.csv")
    return df, movie_features

df, movie_features = load_data()

# Initialize session state
if 'liked_movies' not in st.session_state:
    st.session_state.liked_movies = []

if 'results' not in st.session_state:
    st.session_state.results = None

@st.cache_data
def get_cached_projection(features_df):
    """Cache PCA projection for visualization."""
    return get_pca_projection(features_df)


coords = get_cached_projection(movie_features)

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🎬 Movie Recommendation System")
st.markdown("Find your next favorite movie based on your preferences!")

# ============================================================================
# SIDEBAR - MOVIE SELECTION
# ============================================================================

with st.sidebar:
    st.header("Step 1: Select Your Favorite Movies")
    
    # Search input
    search_query = st.text_input(
        "🔍 Search for a movie:",
        placeholder="Type movie name..."
    )
    
    # Show suggestions
    if search_query and len(search_query) > 1:
        suggestions = get_movie_suggestions(df, search_query)
        
        if suggestions:
            selected_movie = st.selectbox(
                "Select from suggestions:",
                suggestions,
                key="movie_selector"
            )
            
            if st.button("➕ Add Movie"):
                movie_idx = df[df['title'] == selected_movie].index[0]
                
                if selected_movie not in st.session_state.liked_movies:
                    if len(st.session_state.liked_movies) < 5:
                        st.session_state.liked_movies.append(selected_movie)
                        st.success(f"✓ Added: {selected_movie}")
                    else:
                        st.error("⚠️ Maximum 5 movies selected!")
                else:
                    st.warning("Movie already in your list!")
    
    # Display selected movies
    st.markdown("---")
    st.subheader(f"Your Selection ({len(st.session_state.liked_movies)}/5)")
    
    for i, movie in enumerate(st.session_state.liked_movies, 1):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{i}. {movie}")
        with col2:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.liked_movies.pop(i-1)
                st.session_state.results = None
                st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

if len(st.session_state.liked_movies) >= 3:
    st.markdown("---")
    
    # Step 2: Model Selection
    st.header("Step 2: Choose Recommendation Model")
    
    model_choice = st.radio(
        "Select a model:",
        ["Cosine Similarity", "K-Nearest Neighbors (KNN)"]
    )

    n_recommendations = 5
    
    # Generate recommendations
    if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Analyzing your taste and generating recommendations..."):
            # Get indices of liked movies
            liked_indices = [df[df['title'] == m].index[0] for m in st.session_state.liked_movies]
            
            # Analyze user taste
            taste_profile = analyze_user_taste(df, liked_indices)
            
            # Generate recommendations for both models (for comparison visualization)
            cosine_model = CosineSimilarityModel(movie_features)
            cosine_rec_indices, cosine_scores = cosine_model.recommend(liked_indices, n_recommendations)

            knn_model = KNNModel(movie_features)
            knn_rec_indices, knn_scores = knn_model.recommend(liked_indices, n_recommendations)

            # Active model for main recommendations tab
            if model_choice == "Cosine Similarity":
                rec_indices, scores = cosine_rec_indices, cosine_scores
            else:
                rec_indices, scores = knn_rec_indices, knn_scores

            explanations = build_recommendation_explanations(
                df=df,
                movie_features_df=movie_features,
                rec_indices=rec_indices,
                liked_indices=liked_indices,
                scores=scores,
                model_name=model_choice
            )

            st.session_state.results = {
                'model_choice': model_choice,
                'liked_indices': liked_indices,
                'rec_indices': rec_indices,
                'scores': scores,
                'cosine_rec_indices': cosine_rec_indices,
                'cosine_scores': cosine_scores,
                'knn_rec_indices': knn_rec_indices,
                'knn_scores': knn_scores,
                'taste_profile': taste_profile,
                'recs_df': format_recommendations(df, rec_indices, scores),
                'explanations': explanations
            }

    if st.session_state.results is not None:
        results = st.session_state.results
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Recommendations", "Model Visualization", "Why These Movies"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Movies Analyzed", results['taste_profile']['num_movies'])
            with col2:
                st.metric("Avg Rating", f"{results['taste_profile']['avg_rating']:.1f}/10")
            with col3:
                st.metric("Avg Popularity", f"{results['taste_profile']['avg_popularity']:.0f}")

            st.subheader("📊 Your Taste Profile")
            fig = px.pie(
                values=results['taste_profile']['genres'].values,
                names=results['taste_profile']['genres'].index,
                title="Genre Distribution of Your Favorites"
            )
            st.plotly_chart(fig, use_container_width=True)

            mood_col, time_col = st.columns(2)

            with mood_col:
                st.markdown("**Mood themes from your liked movies**")
                top_keywords = results['taste_profile'].get('top_keywords')
                if top_keywords is not None and len(top_keywords) > 0:
                    mood_line = " • ".join([kw.title() for kw in top_keywords.index[:6]])
                    st.write(mood_line)
                else:
                    st.caption("Not enough keyword data to infer mood themes.")

            with time_col:
                st.markdown("**Your time preference**")
                avg_year = results['taste_profile'].get('avg_year')
                favorite_decade = results['taste_profile'].get('favorite_decade')
                if favorite_decade is not None:
                    if avg_year is not None:
                        st.write(f"Mostly {favorite_decade}s films • Average release year: {avg_year:.0f}")
                    else:
                        st.write(f"Mostly {favorite_decade}s films")
                else:
                    st.caption("Release year data is limited for this selection.")

            st.subheader(f"🎯 Recommended Movies ({results['model_choice']})")
            for pos, (_, row) in enumerate(results['recs_df'].iterrows()):
                rec_idx = int(results['rec_indices'][pos])
                exp = results['explanations'][rec_idx]

                with st.container(border=True):
                    poster_col, info_col = st.columns([1, 3])

                    with poster_col:
                        poster_url = row.get('Poster URL')
                        if isinstance(poster_url, str) and poster_url:
                            st.image(poster_url, use_container_width=True)
                        else:
                            st.caption("Poster unavailable")

                    with info_col:
                        st.markdown(f"### {row['Title']}")
                        st.caption(f"Release Date: {row['Release Date']}")

                        score_col1, score_col2 = st.columns(2)
                        with score_col1:
                            st.metric("IMDb Rating", f"{row['IMDb Rating']}/10")
                        with score_col2:
                            st.metric("Match Score", f"{row['similarity_score']}%")

                        st.write(f"**Genres:** {row['Genres']}")
                        st.write(f"**Language:** {row['Language']}")
                        st.write(f"**Description:** {row['Description']}")
                        st.info(exp['summary'])

        with tab2:
            st.subheader("🧭 Model-Specific Visualizations")
            st.caption("Projection uses PCA. Distances shown here are approximate visual aids; recommendations are computed in full feature space.")

            left_col, right_col = st.columns(2)

            with left_col:
                fig_cosine = build_visualization_figure(
                    df=df,
                    movie_features_df=movie_features,
                    coords=coords,
                    liked_indices=results['liked_indices'],
                    rec_indices=results['cosine_rec_indices'],
                    model_type="Cosine Similarity",
                    rec_scores=results['cosine_scores']
                )
                st.plotly_chart(fig_cosine, use_container_width=True)
                st.caption("Cosine view: Orange rays connect the user taste centroid to cosine-selected recommendations.")

            with right_col:
                fig_knn = build_visualization_figure(
                    df=df,
                    movie_features_df=movie_features,
                    coords=coords,
                    liked_indices=results['liked_indices'],
                    rec_indices=results['knn_rec_indices'],
                    model_type="K-Nearest Neighbors (KNN)"
                )
                st.plotly_chart(fig_knn, use_container_width=True)
                st.caption("KNN view: Blue links connect each recommendation to its nearest liked movie.")

            st.markdown("### 🔄 Overlap and Differences")
            cosine_set = set(map(int, results['cosine_rec_indices']))
            knn_set = set(map(int, results['knn_rec_indices']))
            both_set = cosine_set.intersection(knn_set)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Overlap Count", len(both_set))
            with m2:
                st.metric("Cosine-Only", len(cosine_set - knn_set))
            with m3:
                st.metric("KNN-Only", len(knn_set - cosine_set))

            overlap_fig = build_overlap_figure(
                df=df,
                coords=coords,
                liked_indices=results['liked_indices'],
                cosine_rec_indices=results['cosine_rec_indices'],
                knn_rec_indices=results['knn_rec_indices']
            )
            st.plotly_chart(overlap_fig, use_container_width=True)

        with tab3:
            st.subheader("🔍 Why each movie was recommended")
            for pos, (_, row) in enumerate(results['recs_df'].iterrows()):
                rec_idx = int(results['rec_indices'][pos])
                exp = results['explanations'][rec_idx]

                with st.container(border=True):
                    st.markdown(f"### {row['Title']}")
                    st.write(f"**Model score:** {exp['model_score']}%")
                    st.write(f"**Closest liked movie:** {exp['nearest_title']} ({exp['nearest_similarity']}% similarity)")

                    shared_genres_text = ', '.join(exp['shared_genres']) if exp['shared_genres'] else 'None detected'
                    shared_keywords_text = ', '.join(exp['shared_keywords']) if exp['shared_keywords'] else 'None detected'

                    st.write(f"**Shared genres:** {shared_genres_text}")
                    st.write(f"**Shared keywords:** {shared_keywords_text}")
                    st.write(f"**Why chosen:** {exp['summary']}")
            
else:
    # Placeholder when not enough movies selected
    st.info(f"👉 Select at least 3 favorite movies to get started! ({len(st.session_state.liked_movies)}/3)")
    
    st.markdown("""
    ### How it works:
    1. **Search & Select:** Find your favorite movies using the search bar
    2. **Choose Model:** Pick between Cosine Similarity or KNN
    3. **Get Recommendations:** See personalized movie suggestions with analysis!
    """)
