import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import string
import os
import logging
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download stopwords
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Initialize session state for bookmarks
if 'bookmarks' not in st.session_state:
    st.session_state.bookmarks = []

# List of minor ingredients to exclude
MINOR_INGREDIENTS = {
    'salt', 'pepper', 'oil', 'butter', 'sugar', 'water', 'vinegar', 'spices', 
    'seasoning', 'herbs', 'parsley', 'cilantro', 'basil', 'thyme', 'rosemary', 
    'oregano', 'bay leaf', 'garlic powder', 'onion powder', 'paprika', 'chili powder'
}

# Non-vegetarian keywords (whole words to avoid false positives)
NON_VEG_KEYWORDS = {
    'chicken', 'beef', 'pork', 'lamb', 'fish', 'shrimp', 'turkey', 
    'bacon', 'sausage', 'ham', 'duck', 'crab', 'lobster', 'gelatin'
}

# Load dataset
@st.cache_data
def load_data():
    start_time = time.time()
    try:
        df = pd.read_csv("food_dataset.csv")
        # Define required and optional columns
        required_columns = ['Ingredients', 'Title']
        optional_columns = ['Image_URL', 'Is_Vegetarian', 'Calories', 'Protein', 'Instructions']
        
        # Check for required columns
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            st.error(f"Required columns missing from dataset: {missing_required}")
            return pd.DataFrame()
        
        # Drop rows with NaN in required columns
        df.dropna(subset=required_columns, inplace=True)
        
        # Handle optional columns
        for col in optional_columns:
            if col not in df.columns:
                if col == 'Image_URL':
                    df[col] = ''
                elif col == 'Is_Vegetarian':
                    df[col] = True
                elif col == 'Calories':
                    df[col] = 'Unknown'
                elif col == 'Protein':
                    df[col] = 'Unknown'
                elif col == 'Instructions':
                    df[col] = 'No instructions available'
        
        # Refine Is_Vegetarian based on keywords
        def check_non_veg(row):
            ingredients = str(row['Ingredients']).lower()
            title = str(row['Title']).lower()
            # Use word boundaries to avoid false positives (e.g., "eggplant" vs "egg")
            for keyword in NON_VEG_KEYWORDS:
                if re.search(r'\b' + re.escape(keyword) + r'\b', ingredients) or \
                   re.search(r'\b' + re.escape(keyword) + r'\b', title):
                    logger.debug(f"Non-veg keyword '{keyword}' found in {row['Title']}")
                    return False
            # Use dataset's Is_Vegetarian if no non-veg keywords are found
            return row['Is_Vegetarian'] if pd.notna(row['Is_Vegetarian']) else True
        
        df['Is_Vegetarian'] = df.apply(check_non_veg, axis=1)
        
        logger.debug(f"Dataset loaded and processed in {time.time() - start_time:.2f} seconds")
        return df
    except FileNotFoundError:
        st.error("food_dataset.csv not found. Please ensure the file exists in the correct directory.")
        return pd.DataFrame()

# Clean text for vectorization
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Filter main ingredients for display
def filter_main_ingredients(ingredients):
    ingredients = str(ingredients).lower()
    ingredient_list = [ing.strip() for ing in ingredients.split(',')]
    main_ingredients = [ing for ing in ingredient_list if ing not in MINOR_INGREDIENTS]
    return ', '.join(main_ingredients) or ingredients

# Compute TF-IDF matrix
@st.cache_data
def vectorize_ingredients(df):
    start_time = time.time()
    df["Cleaned_Ingredients"] = df["Ingredients"].apply(clean_text)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(df["Cleaned_Ingredients"])
    logger.debug(f"TF-IDF vectorization completed in {time.time() - start_time:.2f} seconds")
    return vectorizer, vectors

# Recommend recipes based on ingredient input
def recommend_recipes(user_input, vectorizer, vectors, df, top_n=5):
    cleaned_input = clean_text(user_input)
    user_vec = vectorizer.transform([cleaned_input])
    sim_scores = cosine_similarity(user_vec, vectors).flatten()
    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    return df.iloc[top_indices], sim_scores[top_indices]

# Simulate image-to-ingredient matching
def image_to_ingredients(image_file):
    filename = image_file.name.lower()
    ingredients = re.sub(r'[^\w\s]', ' ', filename).split()
    ingredients = [ing for ing in ingredients if ing not in MINOR_INGREDIENTS and ing not in stop_words]
    return ', '.join(ingredients) if ingredients else ""

# Streamlit App
st.set_page_config(page_title="Recipe Recommender", page_icon="🍲", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stTextInput > div > div > input {border-radius: 10px; padding: 10px;}
    .recipe-card {background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;}
    .veg-badge {background-color: #28a745; color: white; padding: 5px 10px; border-radius: 12px; font-size: 12px;}
    .non-veg-badge {background-color: #dc3545; color: white; padding: 5px 10px; border-radius: 12px; font-size: 12px;}
    .bookmark-btn {background-color: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer;}
    .bookmark-btn:hover {background-color: #0056b3;}
    .nutrition-info {font-size: 14px; color: #555;}
    .find-btn {background-color: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer;}
    .find-btn:hover {background-color: #218838;}
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation and filters
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Go to", ["Recipe Search", "My Bookmarks"])

    st.markdown("### Filters")
    show_veg = st.checkbox("Vegetarian Only", value=True)
    show_non_veg = st.checkbox("Non-Vegetarian Only", value=True)

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

# Main content
if page == "Recipe Search":
    st.title("🍽️ Recipe Recommender")
    st.markdown("Discover delicious recipes based on ingredients or images!")

    # Input section
    with st.container():
        st.markdown("### Enter Your Ingredients")
        user_input = st.text_input(
            "Ingredients (comma-separated):",
            placeholder="e.g., tomato, onion, garlic",
            help="Type the ingredients you have, separated by commas."
        )
        
        st.markdown("### Upload Ingredient Images")
        uploaded_images = st.file_uploader(
            "Upload images of ingredients (e.g., tomato.jpg):",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload images to detect ingredients (based on filename)."
        )

        # Process uploaded images
        image_ingredients = []
        if uploaded_images:
            for img_file in uploaded_images:
                ingredients = image_to_ingredients(img_file)
                if ingredients:
                    image_ingredients.append(ingredients)
            if image_ingredients:
                st.write("Detected ingredients from images:", ', '.join(image_ingredients))
        
        # Combine text and image inputs
        all_ingredients = [user_input] + image_ingredients
        combined_input = ', '.join([ing for ing in all_ingredients if ing]).strip()
        
        find_button = st.button("Find Recipes", key="find_recipes", help="Click to find recipes based on your inputs")

    # Display recommended recipes
    if combined_input and find_button:
        start_time = time.time()
        df = load_data()
        logger.debug(f"Data loaded for search in {time.time() - start_time:.2f} seconds")
        if df.empty:
            st.warning("No data available to display recipes.")
        else:
            # Apply filters
            if show_veg and not show_non_veg:
                df = df[df['Is_Vegetarian'] == True]
            elif show_non_veg and not show_veg:
                df = df[df['Is_Vegetarian'] == False]
            elif not show_veg and not show_non_veg:
                df = pd.DataFrame()

            if not df.empty:
                vectorizer, vectors = vectorize_ingredients(df)
                results, scores = recommend_recipes(combined_input, vectorizer, vectors, df)

                st.markdown("### Recommended Recipes")
                for i, (index, row) in enumerate(results.iterrows()):
                    with st.container():
                        st.markdown(f"<div class='recipe-card'>", unsafe_allow_html=True)
                        badge_class = "veg-badge" if row['Is_Vegetarian'] else "non-veg-badge"
                        badge_text = "Vegetarian" if row['Is_Vegetarian'] else "Non-Vegetarian"
                        st.markdown(f"**{i+1}. {row['Title']}** <span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Ingredients**: {filter_main_ingredients(row['Ingredients'])}")
                        st.markdown(f"<div class='nutrition-info'>**Calories**: {row['Calories']} | **Protein**: {row['Protein']}</div>", unsafe_allow_html=True)
                        if 'Instructions' in row:
                            st.markdown(f"**Instructions**: {row['Instructions'][:300]}...")
                        # Bookmark button
                        if st.button(f"Bookmark {row['Title']}", key=f"bookmark_{index}"):
                            if row['Title'] not in st.session_state.bookmarks:
                                st.session_state.bookmarks.append(row['Title'])
                                st.success(f"Added {row['Title']} to bookmarks!")
                        if debug_mode:
                            st.write(f"Image Path for {row['Title']}: {row['Image_URL']}")
                            st.write(f"Is_Vegetarian: {row['Is_Vegetarian']}")
                            # Show which keyword triggered non-veg, if any
                            ingredients = str(row['Ingredients']).lower()
                            title = str(row['Title']).lower()
                            triggered_keywords = [kw for kw in NON_VEG_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', ingredients) or re.search(r'\b' + re.escape(kw) + r'\b', title)]
                            if triggered_keywords:
                                st.write(f"Non-veg keywords found: {', '.join(triggered_keywords)}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("---")
                logger.debug(f"Recipes displayed in {time.time() - start_time:.2f} seconds")
            else:
                st.warning("No recipes match the selected filters.")

# Bookmarks page
elif page == "My Bookmarks":
    st.title("📚 My Bookmarked Recipes")
    if st.session_state.bookmarks:
        for bookmark in st.session_state.bookmarks:
            st.markdown(f"- {bookmark}")
            if st.button(f"Remove {bookmark}", key=f"remove_{bookmark}"):
                st.session_state.bookmarks.remove(bookmark)
                st.success(f"Removed {bookmark} from bookmarks!")
                st.experimental_rerun()
    else:
        st.write("No recipes bookmarked yet.")
