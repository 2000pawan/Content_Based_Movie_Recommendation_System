import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# ✅ Load and clean the dataset
df = pd.read_csv("movies_content.csv")
df = df.iloc[:, [-1, 1]]  # Assuming last column = title, 2nd = description
df.columns = ['title', 'description']
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ✅ TF-IDF + Nearest Neighbors model
sw = list(ENGLISH_STOP_WORDS)
tfidf = TfidfVectorizer(lowercase=True, stop_words=sw)
X = tfidf.fit_transform(df['description']).toarray()

model = NearestNeighbors(metric='cosine')
model.fit(X)

# ✅ Placeholder poster for missing ones
DEFAULT_POSTER = "https://via.placeholder.com/300x450.png?text=No+Poster"

# ✅ Fetch poster and plot from OMDb
def fetch_poster_omdb(movie_title):
    api_key = "3a35a37c"
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    response = requests.get(url).json()

    if response.get('Response') == 'True':
        poster = response.get('Poster')
        plot = response.get('Plot', 'No plot available')
        if not poster or poster == "N/A":
            poster = DEFAULT_POSTER
        return poster, plot
    else:
        return DEFAULT_POSTER, "Poster not found or Movie not found"

# ✅ Recommend function
def recommend(movie_title):
    try:
        index = df[df['title'].str.lower() == movie_title.lower()].index[0]
    except IndexError:
        return [], [], []

    distances, indices = model.kneighbors([X[index]], n_neighbors=6)
    recommended_titles = []
    posters = []
    plots = []

    for i in indices[0][1:]:  # Skip the first (the movie itself)
        title = df.iloc[i]['title']
        poster, plot = fetch_poster_omdb(title)
        recommended_titles.append(title)
        posters.append(poster)
        plots.append(plot)

    return recommended_titles, posters, plots

# ✅ Streamlit UI
image = Image.open('img.jpg')
st.image(image, caption='Movie Recommendation System')

st.title('🎬 Movie Recommendation System (Content-Based)')
selected_movie = st.selectbox("Type or select a movie:", df['title'].values)

if st.button('Show Recommendation'):
    names, posters, plots = recommend(selected_movie)

    if not names:
        st.warning("Movie not found in database.")
    else:
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.markdown(
                    f"<img src='{posters[i]}' width='100%' style='border-radius: 10px;'>",
                    unsafe_allow_html=True
                )
                st.text(names[i])
                # st.caption(plots[i])

# ✅ Footer
st.markdown(
    "<h6 style='text-align:center; color: white;'>Developed by: PAWAN YADAV © 2023</h6>",
    unsafe_allow_html=True
)
