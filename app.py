# Complete working code for your Recipe Recommender

import streamlit as st
import pandas as pd
import numpy as np
import nltk
import os
import re
from PIL import Image
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Download NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

@st.cache_resource
def load_resnet_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

@st.cache_data
def load_data():
    df = pd.read_csv("food_dataset.csv")
    if 'Cleaned_Ingredients' not in df.columns:
        df['Cleaned_Ingredients'] = df['Ingredients'].apply(lambda x: preprocess_ingredients(eval(x) if isinstance(x, str) else x))
    return df

def preprocess_ingredients(ingredients):
    ingredients = [re.sub(r'[^a-zA-Z\s]', '', ing.lower()).strip() for ing in ingredients]
    ingredients = [lemmatizer.lemmatize(ing) for ing in ingredients if ing]
    return ' '.join(ingredients)

def get_image_features(image_path, model):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img = img.convert("RGB")
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
    except Exception:
        return None

def classify_image(uploaded_file, df, model):
    try:
        img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        input_features = model.predict(img_array).flatten()

        similarities = []
        for _, row in df.iterrows():
            img_path = os.path.join("Food Images", row["Image_Name"])
            if not os.path.exists(img_path):
                similarities.append(0)
                continue
            features = get_image_features(img_path, model)
            sim = cosine_similarity([input_features], [features])[0][0] if features is not None else 0
            similarities.append(sim)

        df["Image_Similarity"] = similarities
        best_match_idx = df["Image_Similarity"].idxmax()
        return df.loc[best_match_idx]["Ingredients"]
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def suggest_substitutes(missing_ingredients):
    subs = {
        'butter': ['margarine', 'olive oil'],
        'milk': ['almond milk', 'soy milk'],
        'egg': ['applesauce', 'banana'],
        'flour': ['almond flour', 'oat flour'],
        'chicken': ['tofu', 'tempeh'],
        'sugar': ['honey', 'maple syrup'],
        'onion': ['shallot', 'leek']
    }
    return {i: subs.get(i.lower(), ['No substitute available']) for i in missing_ingredients}

def scale_ingredients(ingredients, orig_servings=2, new_servings=2):
    scaled = []
    try:
        scale = new_servings / orig_servings
        for ing in ingredients:
            match = re.match(r"(\d+\.?\d*)\s+(.+)", ing)
            if match:
                qty, rest = float(match.group(1)), match.group(2)
                scaled.append(f"{qty * scale:.1f} {rest}")
            else:
                scaled.append(ing)
    except:
        return ingredients
    return scaled

# -------------------------- Streamlit App -------------------------- #
st.set_page_config(page_title="Recipe Recommender", layout="wide")

st.markdown("""
<style>
.header { background: #4CAF50; color: white; padding: 10px; border-radius: 8px; text-align: center; }
.recipe-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h2>🍽️ Recipe Recommender</h2><p>Upload an image or input ingredients to get recipe suggestions!</p></div>', unsafe_allow_html=True)

df = load_data()
model = load_resnet_model()

with st.sidebar:
    st.header("Filter Recipes")
    vegetarian = st.checkbox("Vegetarian Only", False)
    servings = st.slider("Servings", 1, 10, 2)
    sort_by = st.selectbox("Sort By", ["Similarity", "Prep Time", "Calories"])
    max_time = st.slider("Max Prep Time (minutes)", 5, 120, 60) if 'Prep_Time' in df.columns else float('inf')
    cuisine = st.multiselect("Cuisine", ['All'] + list(df['Cuisine'].unique()), default=['All']) if 'Cuisine' in df.columns else ['All']

st.subheader("Enter Ingredients")
method = st.radio("Choose Input Method:", ["Text Input", "Image Upload"])

ingredients_list = []

if method == "Text Input":
    user_input = st.text_area("Enter ingredients (comma-separated):")
    if user_input:
        ingredients_list = [x.strip() for x in user_input.split(',')]
else:
    uploaded_file = st.file_uploader("Upload an image of your ingredient(s)", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)
        ingredients = classify_image(uploaded_file, df, model)
        if ingredients:
            ingredients_list = eval(ingredients) if isinstance(ingredients, str) else ingredients
            st.success(f"Detected Ingredients: {', '.join(ingredients_list)}")

if st.button("Find Recipes"):
    if not ingredients_list:
        st.warning("Please enter ingredients or upload an image.")
    else:
        processed_input = preprocess_ingredients(ingredients_list)
        vectorizer = TfidfVectorizer()
        recipe_vectors = vectorizer.fit_transform(df["Cleaned_Ingredients"])
        input_vector = vectorizer.transform([processed_input])
        df["Similarity"] = cosine_similarity(input_vector, recipe_vectors).flatten()

        results = df.copy()
        if vegetarian:
            nonveg = ['chicken','beef','pork','fish','shrimp','lamb','bacon','ham','meat','turkey']
            results = results[~results['Ingredients'].apply(lambda x: any(i in str(x).lower() for i in nonveg))]
        if 'Cuisine' in df.columns and 'All' not in cuisine:
            results = results[results['Cuisine'].isin(cuisine)]
        if 'Prep_Time' in df.columns:
            results = results[results['Prep_Time'] <= max_time]

        results = results.sort_values(by=sort_by if sort_by in results.columns else 'Similarity', ascending=True if sort_by in ['Prep Time', 'Calories'] else False)

        if results.empty:
            st.warning("No matching recipes found!")
        else:
            for _, row in results.head(5).iterrows():
                with st.container():
                    st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                    st.markdown(f"### {row['Title']}")
                    ing = eval(row["Ingredients"]) if isinstance(row["Ingredients"], str) else row["Ingredients"]
                    scaled = scale_ingredients(ing, 2, servings)
                    st.markdown(f"**Ingredients (for {servings} servings):** {', '.join(scaled)}")

                    # Missing ingredient analysis
                    missing = set(i.lower() for i in ing) - set(i.lower() for i in ingredients_list)
                    if missing:
                        subs = suggest_substitutes(missing)
                        st.markdown("**Missing Ingredients & Substitutes:**")
                        for m, s in subs.items():
                            st.write(f"- {m}: {', '.join(s)}")

                    st.markdown(f"**Instructions:** {row['Instructions']}")
                    if 'Prep_Time' in row:
                        st.write(f"⏱️ Prep Time: {row['Prep_Time']} mins")
                    if 'Calories' in row:
                        st.write(f"🔥 Calories: {row['Calories']} kcal")

                    image_path = os.path.join("Food Images", row['Image_Name'])
                    if os.path.exists(image_path):
                        st.image(image_path, caption=row['Title'], use_column_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)
