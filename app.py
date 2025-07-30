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
logging.basicConfig(level=logging.DEBUG)
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
        required_columns = ['Ingredients', 'Title', 'Image_Name']
        optional_columns = ['Is_Vegetarian', 'Calories', 'Protein', 'Instructions']
        
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
                if col == 'Is_Vegetarian':
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
            for keyword in NON_VEG_KEYWORDS:
                if re.search(r'\b' + re.escape(keyword) + r'\b', ingredients) or \
                   re.search(r'\b' + re.escape(keyword) + r'\b', title):
                    logger.debug(f"Non-veg keyword '{keyword}' found in {row['Title']}")
                    return False
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
    .main {background-color: #f8f9fa;}
    .stTextInput > div > div > input {border-radius: 10px; padding: 10px;}
    .recipe-card {background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;}
    .veg-badge {background-color: #28a745; color: white; padding: 5px 10px; border-radius: 12px; font-size: 12px;}
    .non-veg-badge {background-color: #dc3545; color: white; padding: 5px 10px; border-radius: 12px; font-size: 12px;}
    .bookmark-btn {background-color: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer;}
    .bookmark-btn:hover {background-color: #0056b3;}
    .use-recipe-btn {background-color: #28a745; color: white; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer;}
    .use-recipe-btn:hover {background-color: #218838;}
    .back-btn {background-color: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer;}
    .back-btn:hover {background-color: #5a6268;}
    .nutrition-info {font-size: 14px; color: #555;}
    .find-btn {background-color: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer;}
    .find-btn:hover {background-color: #218838;}
    .recipe-image {max-width: 300px; border-radius: 10px; margin-bottom: 10px;}
    .delete-btn {background-color: #dc3545; color: white; border: none; padding: 6px 12px; border-radius: 8px; cursor: pointer;}
    .delete-btn:hover {background-color: #c82333;}
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation and filters
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Go to", ["Recipe Search", "My Bookmarks", "Search History"])

    if page == "Recipe Search":
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

    # Store search in history
    if combined_input and find_button:
        if combined_input not in st.session_state.search_history:
            st.session_state.search_history.append(combined_input)
        
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
                st.session_state.search_results = results
                st.session_state.search_scores = scores
            else:
                st.session_state.search_results = None
                st.session_state.search_scores = None
                st.warning("No recipes match the selected filters.")

    # Display recipe details if a recipe is selected
    if st.session_state.selected_recipe is not None:
        row = st.session_state.selected_recipe
        with st.container():
            st.markdown(f"<div class='recipe-card'>", unsafe_allow_html=True)
            badge_class = "veg-badge" if row['Is_Vegetarian'] else "non-veg-badge"
            badge_text = "Vegetarian" if row['Is_Vegetarian'] else "Non-Vegetarian"
            st.markdown(f"**{row['Title']}** <span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
            
            # Load and display image
            image_name = row['Image_Name']
            image = load_recipe_image(image_name)
            if image:
                st.image(image, caption=row['Title'], width=300, use_container_width=False, output_format="JPEG", clamp=True)
            else:
                st.warning(f"Image not found for {row['Title']}")
            
            st.markdown(f"**Ingredients**: {filter_main_ingredients(row['Ingredients'])}")
            st.markdown(f"<div class='nutrition-info'>**Calories**: {row['Calories']} | **Protein**: {row['Protein']}</div>", unsafe_allow_html=True)
            if 'Instructions' in row:
                st.markdown(f"**Instructions**: {row['Instructions']}")
            if st.button("Back to Results", key=f"back_{row['Title']}", help="Return to search results"):
                st.session_state.selected_recipe = None
                st.rerun()
            if debug_mode:
                st.write(f"Image Path for {row['Title']}: Food Images/{image_name}.jpg")
                st.write(f"Is_Vegetarian: {row['Is_Vegetarian']}")
                ingredients = str(row['Ingredients']).lower()
                title = str(row['Title']).lower()
                triggered_keywords = [kw for kw in NON_VEG_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', ingredients) or re.search(r'\b' + re.escape(kw) + r'\b', title)]
                if triggered_keywords:
                    st.write(f"Non-veg keywords found: {', '.join(triggered_keywords)}")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Display recommended recipes
    elif st.session_state.search_results is not None:
        st.markdown("### Recommended Recipes")
        for i, (index, row) in enumerate(st.session_state.search_results.iterrows()):
            with st.container():
                st.markdown(f"<div class='recipe-card'>", unsafe_allow_html=True)
                badge_class = "veg-badge" if row['Is_Vegetarian'] else "non-veg-badge"
                badge_text = "Vegetarian" if row['Is_Vegetarian'] else "Non-Vegetarian"
                st.markdown(f"**{i+1}. {row['Title']}** <span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
                
                # Load and display image
                image_name = row['Image_Name']
                image = load_recipe_image(image_name)
                if image:
                    st.image(image, caption=row['Title'], width=300, use_container_width=False, output_format="JPEG", clamp=True)
                else:
                    st.warning(f"Image not found for {row['Title']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use Recipe", key=f"use_{index}", help=f"View details for {row['Title']}"):
                        st.session_state.selected_recipe = row
                        st.rerun()
                with col2:
                    if st.button("Bookmark", key=f"bookmark_{index}", help=f"Bookmark {row['Title']}"):
                        if row['Title'] not in st.session_state.bookmarks:
                            st.session_state.bookmarks.append(row['Title'])
                            st.success(f"Added {row['Title']} to bookmarks!")
                if debug_mode:
                    st.write(f"Image Path for {row['Title']}: Food Images/{image_name}.jpg")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("---")

# Bookmarks page
elif page == "My Bookmarks":
    st.title("📚 My Bookmarked Recipes")
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
                    # Load and display image
                    image_name = row['Image_Name']
                    image = load_recipe_image(image_name)
                    if image:
                        st.image(image, caption=bookmark, width=300, use_container_width=False, output_format="JPEG", clamp=True)
                    else:
                        st.warning(f"Image not found for {bookmark}")
                    if st.button(f"Remove {bookmark}", key=f"remove_{bookmark}"):
                        st.session_state.bookmarks.remove(bookmark)
                        st.success(f"Removed {bookmark} from bookmarks!")
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.write(f"- {bookmark} (Recipe data not found)")
                if st.button(f"Remove {bookmark}", key=f"remove_{bookmark}"):
                    st.session_state.bookmarks.remove(bookmark)
                    st.success(f"Removed {bookmark} from bookmarks!")
                    st.rerun()
    else:
        st.write("No recipes bookmarked yet.")

# History page
elif page == "Search History":
    st.title("📜 Search History")
    if st.session_state.search_history:
        for i, search in enumerate(st.session_state.search_history):
            with st.container():
                st.markdown(f"<div class='recipe-card'>", unsafe_allow_html=True)
                st.write(f"**Search {i+1}**: {search}")
                if st.button("Delete", key=f"delete_search_{i}", help=f"Delete search: {search}"):
                    st.session_state.search_history.remove(search)
                    st.success(f"Deleted search: {search}")
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.write("No search history yet.")
