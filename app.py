import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import string

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("food_dataset.csv")
    df.dropna(subset=['Ingredients', 'Title'], inplace=True)
    return df

# Clean ingredients text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Compute TF-IDF matrix
@st.cache_data
def vectorize_ingredients(df):
    df["Cleaned_Ingredients"] = df["Ingredients"].apply(clean_text)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(df["Cleaned_Ingredients"])
    return vectorizer, vectors

# Recommend recipes based on ingredient input
def recommend_recipes(user_input, vectorizer, vectors, df, top_n=5):
    cleaned_input = clean_text(user_input)
    user_vec = vectorizer.transform([cleaned_input])
    sim_scores = cosine_similarity(user_vec, vectors).flatten()
    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    return df.iloc[top_indices], sim_scores[top_indices]

# Streamlit App
st.set_page_config(page_title="Recipe Recommender", page_icon="🍲", layout="centered")
st.title("🍽️ Recipe Recommender")
st.write("Enter the ingredients you have. We'll suggest recipes!")

user_input = st.text_input("Ingredients (comma-separated):", placeholder="e.g., tomato, onion, garlic")

if user_input:
    df = load_data()
    vectorizer, vectors = vectorize_ingredients(df)
    results, scores = recommend_recipes(user_input, vectorizer, vectors, df)

    st.subheader("Recommended Recipes:")
    for i, (index, row) in enumerate(results.iterrows()):
        st.markdown(f"**{i+1}. {row['Title']}**")
        st.write(f"**Ingredients**: {row['Ingredients']}")
        if 'Instructions' in row:
            st.write(f"**Instructions**: {row['Instructions'][:300]}...")  # Trim long text
        st.markdown("---")
