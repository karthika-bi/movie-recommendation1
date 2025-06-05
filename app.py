import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")  # Make sure this file is in the same folder or give full path
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    similarity = cosine_similarity(tfidf_matrix)
    return movies, similarity

movies, similarity = load_data()

def recommend(movie_title, n=5):
    matched_movies = movies[movies['title'].str.lower().str.contains(movie_title.lower())]
    if matched_movies.empty:
        return ["Movie not found. Please check the title."]
    idx = matched_movies.index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    recommendations = [movies.iloc[i[0]]['title'] for i in sorted_scores]
    return recommendations

st.title("🎬 Movie Recommendation System")

user_input = st.text_input("Enter a movie title:")

if user_input:
    recommendations = recommend(user_input)
    st.write("### Recommendations:")
    for rec in recommendations:
        st.write("- " + rec)
