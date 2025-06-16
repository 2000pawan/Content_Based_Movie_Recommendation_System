import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# Constants
DEFAULT_POSTER = "https://via.placeholder.com/300x450.png?text=No+Poster"

# Load and preprocess data
df = pd.read_csv("movies_content.csv")
df = df.iloc[:, [-1, 1]]
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Vectorization
stopwords = list(ENGLISH_STOP_WORDS)
tfidf = TfidfVectorizer(lowercase=True, stop_words=stopwords)
X = tfidf.fit_transform(df['description']).toarray()

# Model
model = NearestNeighbors(metric='cosine')
model.fit(X)

# Load image
image = Image.open('img.jpg')
st.image(image, caption='Movie Recommendation System')

# OMDb Poster Fetcher
def fetch_poster_omdb(movie_title):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url).json()
        if response['Response'] == 'True':
            poster = response.get('Poster')
            plot = response.get('Plot', 'No plot available')
            if poster == "N/A" or poster is None:
                poster = DEFAULT_POSTER
            return poster, plot
    except:
        pass
    return DEFAULT_POSTER, "Poster not found or Movie not found."

# Recommendation logic
def recommend(movie_name):
    try:
        index = df[df['name'].str.lower() == movie_name.lower()].index[0]
    except IndexError:
        st.error("Movie not found in database.")
        return [], [], []

    distances, indices = model.kneighbors([X[index]], n_neighbors=6)
    recommendations = []
    posters = []
    plots = []

    for i in indices[0][1:]:  # Skip the first one (input movie itself)
        title = df.iloc[i]['name']
        poster, plot = fetch_poster_omdb(title)
        recommendations.append(title)
        posters.append(poster)
        plots.append(plot)

    return recommendations, posters, plots

# UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown:",
    df['name'].values
)

if st.button("Show Recommendation"):
    names, posters, plots = recommend(selected_movie)
    if len(names) == 0:
        st.warning("No recommendations found.")
    else:
        cols = st.columns(5)
        for i in range(min(5, len(names))):
            with cols[i]:
                st.markdown(
                    f"<img src='{posters[i]}' width='100%' style='border-radius: 10px;'>",
                    unsafe_allow_html=True
                )
                st.text(names[i])
                # st.caption(plots[i])

# Footer
st.markdown(
    "<hr><h6 style='text-align:center;'>Developed by: PAWAN YADAV © 2023 All rights reserved.</h6>",
    unsafe_allow_html=True
)
