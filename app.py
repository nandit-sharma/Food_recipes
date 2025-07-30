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
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download stopwords
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Initialize session state
if 'bookmarks' not in st.session_state:
    st.session_state.bookmarks = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'selected_recipe' not in st.session_state:
    st.session_state.selected_recipe = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'search_scores' not in st.session_state:
    st.session_state.search_scores = None
if 'liked_recipes' not in st.session_state:
    st.session_state.liked_recipes = []
if 'viewed_recipes' not in st.session_state:
    st.session_state.viewed_recipes = []

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
        required_columns = ['Ingredients', 'Title', 'Image_Name']
        optional_columns = ['Is_Vegetarian', 'Calories', 'Protein', 'Instructions']
        
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            st.error(f"Required columns missing from dataset: {missing_required}")
            return pd.DataFrame()
        
        df.dropna(subset=required_columns, inplace=True)
        
        for col in optional_columns:
            if col not in df.columns:
                if col == 'Is_Vegetarian':
                    df[col] = True
                elif col == 'Calories':
                    df[col] = 'Unknown'
                elif col == 'Protein':
                    df[col] = 'Unknown'
                elif col == 'Instructions':
                    df[col] = 'No instructions available'
        
        def check_non_veg(row):
            ingredients = str(row['Ingredients']).lower()
            title = str(row['Title']).lower()
            for keyword in NON_VEG_KEYWORDS:
                if re.search(r'\b' + re.escape(keyword) + r'\b', ingredients) or \
                   re.search(r'\b' + re.escape(keyword) + r'\b', title):
                    logger.info(f"Non-veg keyword '{keyword}' found in {row['Title']}")
                    return False
            return row['Is_Vegetarian'] if pd.notna(row['Is_Vegetarian']) else True
        
        df['Is_Vegetarian'] = df.apply(check_non_veg, axis=1)
        
        logger.info(f"Dataset loaded and processed in {time.time() - start_time:.2f} seconds")
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
    logger.info(f"TF-IDF vectorization completed in {time.time() - start_time:.2f} seconds")
    return vectorizer, vectors

# Recommend recipes based on ingredient input
def recommend_recipes(user_input, vectorizer, vectors, df, top_n=5):
    cleaned_input = clean_text(user_input)
    user_vec = vectorizer.transform([cleaned_input])
    sim_scores = cosine_similarity(user_vec, vectors).flatten()
    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    return df.iloc[top_indices], sim_scores[top_indices]

# Personalized recipe recommendation based on user interactions
def personalized_recommendations(df, vectorizer, vectors, top_n=5):
    interacted_recipes = list(set(st.session_state.liked_recipes + st.session_state.bookmarks + st.session_state.viewed_recipes))
    if not interacted_recipes:
        return pd.DataFrame(), np.array([])
    
    interacted_indices = df[df['Title'].isin(interacted_recipes)].index
    if len(interacted_indices) == 0:
        return pd.DataFrame(), np.array([])
    
    interacted_vectors = vectors[interacted_indices]
    avg_vector = np.asarray(np.mean(interacted_vectors, axis=0))
    sim_scores = cosine_similarity(avg_vector.reshape(1, -1), vectors).flatten()
    
    sim_scores[interacted_indices] = -1
    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    return df.iloc[top_indices], sim_scores[top_indices]

# Load image from Food Images directory
def load_recipe_image(image_name):
    try:
        image_path = os.path.join("Food Images", f"{image_name}.jpg")
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            logger.warning(f"Image not found: {image_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading image {image_name}: {str(e)}")
        return None

# Streamlit App
st.set_page_config(page_title="Recipe Recommender", page_icon="🍲", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    .main {background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); font-family: 'Roboto', sans-serif;}
    .stTextInput > div > div > input {
        border-radius: 12px; 
        padding: 12px; 
        border: 1px solid #ced4da; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #28a745; 
        box-shadow: 0 0 8px rgba(40, 167, 69, 0.3);
    }
    .recipe-card {
        background: white; 
        padding: 24px; 
        border-radius: 16px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
        margin-bottom: 24px; 
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .recipe-card:hover {
        transform: translateY(-4px); 
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    .veg-badge {
        background-color: #28a745; 
        color: white; 
        padding: 6px 12px; 
        border-radius: 16px; 
        font-size: 13px; 
        font-weight: 500;
    }
    .non-veg-badge {
        background-color: #dc3545; 
        color: white; 
        padding: 6px 12px; 
        border-radius: 16px; 
        font-size: 13px; 
        font-weight: 500;
    }
    .action-btn {
        padding: 10px 20px; 
        border-radius: 10px; 
        border: none; 
        font-weight: 500; 
        cursor: pointer; 
        transition: transform 0.2s, opacity 0.2s;
        margin-right: 8px;
    }
    .action-btn:hover {
        transform: scale(1.05); 
        opacity: 0.9;
    }
    .use-recipe-btn {
        background-color: #28a745; 
        color: white;
    }
    .use-recipe-btn:hover {
        background-color: #218838;
    }
    .bookmark-btn {
        background-color: #007bff; 
        color: white;
    }
    .bookmark-btn:hover {
        background-color: #0056b3;
    }
    .like-btn {
        background-color: #fd7e14; 
        color: white;
    }
    .like-btn:hover {
        background-color: #e06c00;
    }
    .unlike-btn {
        background-color: #dc3545; 
        color: white;
    }
    .unlike-btn:hover {
        background-color: #c82333;
    }
    .back-btn {
        background-color: #6c757d; 
        color: white;
    }
    .back-btn:hover {
        background-color: #5a6268;
    }
    .nutrition-info {
        font-size: 14px; 
        color: #495057; 
        margin-top: 12px;
    }
    .find-btn {
        background-color: #28a745; 
        color: white; 
        padding: 12px 24px; 
        border-radius: 10px; 
        border: none; 
        font-weight: 500; 
        cursor: pointer; 
        transition: transform 0.2s, opacity 0.2s;
    }
    .find-btn:hover {
        transform: scale(1.05); 
        opacity: 0.9;
    }
    .delete-btn {
        background-color: #dc3545; 
        color: white; 
        padding: 8px 16px; 
        border-radius: 10px; 
        border: none; 
        font-weight: 500; 
        cursor: pointer; 
        transition: transform 0.2s, opacity 0.2s;
    }
    .delete-btn:hover {
        transform: scale(1.05); 
        opacity: 0.9;
    }
    .sidebar .stRadio > div {
        background: white; 
        padding: 16px; 
        border-radius: 12px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .sidebar .stRadio > div > label {
        font-weight: 500; 
        color: #343a40; 
        padding: 8px 12px; 
        border-radius: 8px; 
        transition: background-color 0.3s;
    }
    .sidebar .stRadio > div > label:hover {
        background-color: #e9ecef;
    }
    .sidebar .stCheckbox > label {
        font-weight: 500; 
        color: #343a40;
    }
    .recipe-image {
        max-width: 320px; 
        border-radius: 12px; 
        margin-bottom: 16px;
    }
    h1, h3 {
        color: #343a40; 
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation and filters
with st.sidebar:
    st.markdown("### 🍴 Navigation")
    page = st.radio("", ["Recipe Search", "Personalized Recommendations", "Liked Recipes", "My Bookmarks", "Search History"],
                   format_func=lambda x: f"{' ' if x == 'Recipe Search' else ' ' if x == 'Personalized Recommendations' else ' ' if x == 'Liked Recipes' else ' ' if x == 'My Bookmarks' else ' '} {x}")

    if page == "Recipe Search":
        st.markdown("### 🛠 Filters")
        show_veg = st.checkbox("Vegetarian Only", value=True, help="Show only vegetarian recipes")
        show_non_veg = st.checkbox("Non-Vegetarian Only", value=True, help="Show only non-vegetarian recipes")

# Main content
# Display recipe details if a recipe is selected (moved to top level for consistency across pages)
if st.session_state.selected_recipe is not None:
    row = st.session_state.selected_recipe
    with st.container():
        st.markdown(f"<div class='recipe-card'>", unsafe_allow_html=True)
        badge_class = "veg-badge" if row['Is_Vegetarian'] else "non-veg-badge"
        badge_text = "Vegetarian" if row['Is_Vegetarian'] else "Non-Vegetarian"
        st.markdown(f"**{row['Title']}** <span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
        
        image_name = row['Image_Name']
        image = load_recipe_image(image_name)
        if image:
            st.image(image, caption=row['Title'], width=320, use_container_width=False, output_format="JPEG", clamp=True)
        else:
            st.warning(f"Image not found for {row['Title']}")
        
        st.markdown(f"**Ingredients**: {filter_main_ingredients(row['Ingredients'])}")
        st.markdown(f"<div class='nutrition-info'>**Calories**: {row['Calories']} | **Protein**: {row['Protein']}</div>", unsafe_allow_html=True)
        if 'Instructions' in row:
            st.markdown(f"**Instructions**: {row['Instructions']}")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Back to Results", key=f"back_{row['Title']}", help="Return to previous page", type="secondary"):
                st.session_state.selected_recipe = None
                st.rerun()
        with col2:
            if row['Title'] in st.session_state.liked_recipes:
                if st.button("Unlike", key=f"unlike_detail_{row['Title']}", help=f"Remove {row['Title']} from liked recipes", type="secondary"):
                    st.session_state.liked_recipes.remove(row['Title'])
                    st.success(f"Unliked {row['Title']}!")
                    st.rerun()
            else:
                if st.button("Like", key=f"like_detail_{row['Title']}", help=f"Like {row['Title']}", type="secondary"):
                    st.session_state.liked_recipes.append(row['Title'])
                    st.success(f"Liked {row['Title']}!")
        st.markdown("</div>", unsafe_allow_html=True)

# Page-specific content
elif page == "Recipe Search":
    st.title("🍽️ Recipe Recommender")
    st.markdown("Discover delicious recipes based on ingredients!")

    # Input section
    with st.container():
        st.markdown("### Enter Your Ingredients")
        user_input = st.text_input(
            "Ingredients (comma-separated):",
            placeholder="e.g., tomato, onion, garlic",
            help="Type the ingredients you have, separated by commas."
        )
        
        find_button = st.button("Find Recipes", key="find_recipes", help="Click to find recipes based on your inputs", type="primary")

    # Store search in history
    if user_input and find_button:
        if user_input not in st.session_state.search_history:
            st.session_state.search_history.append(user_input)
        
        start_time = time.time()
        df = load_data()
        logger.info(f"Data loaded for search in {time.time() - start_time:.2f} seconds")
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
                results, scores = recommend_recipes(user_input, vectorizer, vectors, df)
                st.session_state.search_results = results
                st.session_state.search_scores = scores
            else:
                st.session_state.search_results = None
                st.session_state.search_scores = None
                st.warning("No recipes match the selected filters.")

    # Display recommended recipes
    if st.session_state.search_results is not None:
        st.markdown("### Recommended Recipes")
        for i, (index, row) in enumerate(st.session_state.search_results.iterrows()):
            with st.container():
                st.markdown(f"<div class='recipe-card'>", unsafe_allow_html=True)
                badge_class = "veg-badge" if row['Is_Vegetarian'] else "non-veg-badge"
                badge_text = "Vegetarian" if row['Is_Vegetarian'] else "Non-Vegetarian"
                st.markdown(f"**{i+1}. {row['Title']}** <span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
                
                image_name = row['Image_Name']
                image = load_recipe_image(image_name)
                if image:
                    st.image(image, caption=row['Title'], width=320, use_container_width=False, output_format="JPEG", clamp=True)
                else:
                    st.warning(f"Image not found for {row['Title']}")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("Use Recipe", key=f"use_{index}", help=f"View details for {row['Title']}", type="primary"):
                        st.session_state.viewed_recipes.append(row['Title'])
                        st.session_state.selected_recipe = row
                        st.rerun()
                with col2:
                    if st.button("Bookmark", key=f"bookmark_{index}", help=f"Bookmark {row['Title']}", type="secondary"):
                        if row['Title'] not in st.session_state.bookmarks:
                            st.session_state.bookmarks.append(row['Title'])
                            st.success(f"Added {row['Title']} to bookmarks!")
                with col3:
                    if row['Title'] in st.session_state.liked_recipes:
                        if st.button("Unlike", key=f"unlike_{index}", help=f"Remove {row['Title']} from liked recipes", type="secondary"):
                            st.session_state.liked_recipes.remove(row['Title'])
                            st.success(f"Unliked {row['Title']}!")
                            st.rerun()
                    else:
                        if st.button("Like", key=f"like_{index}", help=f"Like {row['Title']}", type="secondary"):
                            st.session_state.liked_recipes.append(row['Title'])
                            st.success(f"Liked {row['Title']}!")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("---")

elif page == "Personalized Recommendations":
    st.title("🌟 Personalized Recommendations")
    st.markdown("Recipes tailored to your interests based on your likes, bookmarks, and views.")
    
    if st.session_state.liked_recipes or st.session_state.bookmarks or st.session_state.viewed_recipes:
        df = load_data()
        if not df.empty:
            vectorizer, vectors = vectorize_ingredients(df)
            personalized_results, personalized_scores = personalized_recommendations(df, vectorizer, vectors)
            if not personalized_results.empty:
                for i, (index, row) in enumerate(personalized_results.iterrows()):
                    with st.container():
                        st.markdown(f"<div class='recipe-card'>", unsafe_allow_html=True)
                        badge_class = "veg-badge" if row['Is_Vegetarian'] else "non-veg-badge"
                        badge_text = "Vegetarian" if row['Is_Vegetarian'] else "Non-Vegetarian"
                        st.markdown(f"**{i+1}. {row['Title']}** <span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
                        
                        image_name = row['Image_Name']
                        image = load_recipe_image(image_name)
                        if image:
                            st.image(image, caption=row['Title'], width=320, use_container_width=False, output_format="JPEG", clamp=True)
                        else:
                            st.warning(f"Image not found for {row['Title']}")
                        
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            if st.button("Use Recipe", key=f"use_personalized_{index}", help=f"View details for {row['Title']}", type="primary"):
                                st.session_state.viewed_recipes.append(row['Title'])
                                st.session_state.selected_recipe = row
                                st.rerun()
                        with col2:
                            if st.button("Bookmark", key=f"bookmark_personalized_{index}", help=f"Bookmark {row['Title']}", type="secondary"):
                                if row['Title'] not in st.session_state.bookmarks:
                                    st.session_state.bookmarks.append(row['Title'])
                                    st.success(f"Added {row['Title']} to bookmarks!")
                        with col3:
                            if row['Title'] in st.session_state.liked_recipes:
                                if st.button("Unlike", key=f"unlike_personalized_{index}", help=f"Remove {row['Title']} from liked recipes", type="secondary"):
                                    st.session_state.liked_recipes.remove(row['Title'])
                                    st.success(f"Unliked {row['Title']}!")
                                    st.rerun()
                            else:
                                if st.button("Like", key=f"like_personalized_{index}", help=f"Like {row['Title']}", type="secondary"):
                                    st.session_state.liked_recipes.append(row['Title'])
                                    st.success(f"Liked {row['Title']}!")
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("---")
            else:
                st.write("No personalized recommendations available. Try liking, bookmarking, or viewing some recipes!")
        else:
            st.warning("No data available to generate recommendations.")
    else:
        st.write("No interactions yet. Like, bookmark, or view some recipes to get personalized suggestions!")

elif page == "Liked Recipes":
    st.title("❤️ Liked Recipes")
    st.markdown("View all the recipes you've liked.")
    if st.session_state.liked_recipes:
        df = load_data()
        for liked_recipe in st.session_state.liked_recipes:
            recipe = df[df['Title'] == liked_recipe]
            if not recipe.empty:
                row = recipe.iloc[0]
                with st.container():
                    st.markdown(f"<div class='recipe-card'>", unsafe_allow_html=True)
                    badge_class = "veg-badge" if row['Is_Vegetarian'] else "non-veg-badge"
                    badge_text = "Vegetarian" if row['Is_Vegetarian'] else "Non-Vegetarian"
                    st.markdown(f"- **{liked_recipe}** <span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
                    image_name = row['Image_Name']
                    image = load_recipe_image(image_name)
                    if image:
                        st.image(image, caption=liked_recipe, width=320, use_container_width=False, output_format="JPEG", clamp=True)
                    else:
                        st.warning(f"Image not found for {liked_recipe}")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("Use Recipe", key=f"use_liked_{liked_recipe}", help=f"View details for {liked_recipe}", type="primary"):
                            st.session_state.viewed_recipes.append(row['Title'])
                            st.session_state.selected_recipe = row
                            st.rerun()
                    with col2:
                        if st.button("Unlike", key=f"unlike_liked_{liked_recipe}", help=f"Remove {liked_recipe} from liked recipes", type="secondary"):
                            st.session_state.liked_recipes.remove(liked_recipe)
                            st.success(f"Unliked {liked_recipe}!")
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.write(f"- {liked_recipe} (Recipe data not found)")
                if st.button("Unlike", key=f"unlike_liked_missing_{liked_recipe}", help=f"Remove {liked_recipe} from liked recipes", type="secondary"):
                    st.session_state.liked_recipes.remove(liked_recipe)
                    st.success(f"Unliked {liked_recipe}!")
                    st.rerun()
    else:
        st.write("No recipes liked yet. Like some recipes to see them here!")

elif page == "My Bookmarks":
    st.title("📚 My Bookmarked Recipes")
    st.markdown("View all your bookmarked recipes.")
    if st.session_state.bookmarks:
        df = load_data()
        for bookmark in st.session_state.bookmarks:
            recipe = df[df['Title'] == bookmark]
            if not recipe.empty:
                row = recipe.iloc[0]
                with st.container():
                    st.markdown(f"<div class='recipe-card'>", unsafe_allow_html=True)
                    badge_class = "veg-badge" if row['Is_Vegetarian'] else "non-veg-badge"
                    badge_text = "Vegetarian" if row['Is_Vegetarian'] else "Non-Vegetarian"
                    st.markdown(f"- **{bookmark}** <span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
                    image_name = row['Image_Name']
                    image = load_recipe_image(image_name)
                    if image:
                        st.image(image, caption=bookmark, width=320, use_container_width=False, output_format="JPEG", clamp=True)
                    else:
                        st.warning(f"Image not found for {bookmark}")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button(f"Use Recipe", key=f"use_bookmark_{bookmark}", help=f"View details for {bookmark}", type="primary"):
                            st.session_state.viewed_recipes.append(row['Title'])
                            st.session_state.selected_recipe = row
                            st.rerun()
                    with col2:
                        if row['Title'] in st.session_state.liked_recipes:
                            if st.button("Unlike", key=f"unlike_bookmark_{bookmark}", help=f"Remove {bookmark} from liked recipes", type="secondary"):
                                st.session_state.liked_recipes.remove(bookmark)
                                st.success(f"Unliked {bookmark}!")
                                st.rerun()
                        else:
                            if st.button("Like", key=f"like_bookmark_{bookmark}", help=f"Like {bookmark}", type="secondary"):
                                st.session_state.liked_recipes.append(bookmark)
                                st.success(f"Liked {bookmark}!")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.write(f"- {bookmark} (Recipe data not found)")
                if st.button(f"Remove", key=f"remove_{bookmark}", help=f"Remove {bookmark} from bookmarks", type="secondary"):
                    st.session_state.bookmarks.remove(bookmark)
                    st.success(f"Removed {bookmark} from bookmarks!")
                    st.rerun()
    else:
        st.write("No recipes bookmarked yet.")

elif page == "Search History":
    st.title("📜 Search History")
    st.markdown("View your recent ingredient searches.")
    if st.session_state.search_history:
        for i, search in enumerate(st.session_state.search_history):
            with st.container():
                st.markdown(f"<div class='recipe-card'>", unsafe_allow_html=True)
                st.write(f"**Search {i+1}**: {search}")
                if st.button("Delete", key=f"delete_search_{i}", help=f"Delete search: {search}", type="secondary"):
                    st.session_state.search_history.remove(search)
                    st.success(f"Deleted search: {search}")
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.write("No search history yet.")
