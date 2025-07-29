import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
import re
import os
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Download NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load ResNet50 model
@st.cache_resource
def load_resnet_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('food_dataset.csv')
    if 'Cleaned_Ingredients' not in df.columns:
        df['Cleaned_Ingredients'] = df['Ingredients'].apply(lambda x: preprocess_ingredients(eval(x) if isinstance(x, str) else x))
    return df

# Preprocess ingredients
def preprocess_ingredients(ingredients):
    ingredients = [re.sub(r'[^a-zA-Z\s]', '', ing.lower()).strip() for ing in ingredients]
    ingredients = [lemmatizer.lemmatize(ing) for ing in ingredients if ing]
    return ' '.join(ingredients)

# Suggest ingredient substitutes
def suggest_substitutes(missing_ingredients):
    substitutes = {
        'butter': ['margarine', 'olive oil'],
        'milk': ['almond milk', 'soy milk'],
        'egg': ['applesauce', 'mashed banana'],
        'flour': ['almond flour', 'oat flour'],
        'chicken': ['tofu', 'tempeh'],
        'cheese': ['nutritional yeast', 'vegan cheese'],
        'onion': ['shallot', 'leek'],
        'garlic': ['garlic powder', 'shallot'],
        'sugar': ['honey', 'maple syrup']
    }
    suggestions = {}
    for ing in missing_ingredients:
        suggestions[ing] = substitutes.get(ing.lower(), ['No substitute available'])
    return suggestions

# Scale ingredients for servings
def scale_ingredients(ingredients, original_servings=2, target_servings=2):
    try:
        scale = target_servings / original_servings
        scaled = []
        for ing in ingredients:
            # Extract quantity if present (e.g., "2 cups flour")
            match = re.match(r'(\d+\.?\d*)\s*(.+)', ing)
            if match:
                quantity, rest = float(match.group(1)), match.group(2)
                scaled_quantity = quantity * scale
                scaled.append(f"{scaled_quantity:.1f} {rest}")
            else:
                scaled.append(ing)  # No quantity, keep as is
        return scaled
    except:
        return ingredients

# Extract image features using ResNet50
def get_image_features(image_path, model):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
    except:
        return None

# Classify uploaded image
def classify_image(uploaded_file, df, model):
    if not uploaded_file:
        return None
    try:
        img = Image.open(uploaded_file)
        img = img.resize((224, 224)).convert('RGB')
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        input_features = model.predict(img_array).flatten()
        
        similarities = []
        for idx, row in df.iterrows():
            img_path = os.path.join("Food Images", row['Image_Name'])
            features = get_image_features(img_path, model)
            if features is not None:
                sim = cosine_similarity([input_features], [features])[0][0]
                similarities.append(sim)
            else:
                similarities.append(0)
        
        df['Image_Similarity'] = similarities
        best_match_idx = df['Image_Similarity'].idxmax()
        return df.loc[best_match_idx]['Ingredients']
    except Exception as e:
        st.error(f"Error processing uploaded image: {str(e)}")
        return None

# Custom CSS for professional UI
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; }
    .stTextInput>label, .stSlider>label, .stCheckbox>label, .stRadio>label, .stMultiselect>label {
        font-weight: bold; color: #333; }
    .recipe-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .header { background-color: #4CAF50; color: white; padding: 10px; border-radius: 8px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Main Streamlit app
def main():
    # Header
    st.markdown('<div class="header"><h1>Recipe Recommender</h1><p>Find delicious recipes based on your ingredients!</p></div>', unsafe_allow_html=True)
    
    # Load dataset and model
    df = load_data()
    model = load_resnet_model()

    # Sidebar for filters
    with st.sidebar:
        st.header("Filters & Settings")
        vegetarian = st.checkbox("Vegetarian Only", value=False)
        sort_by = st.selectbox("Sort By", ["Similarity", "Prep Time" if 'Prep_Time' in df.columns else None, "Calories" if 'Calories' in df.columns else None], index=0)
        if 'Prep_Time' in df.columns:
            max_time = st.slider("Max Prep Time (minutes)", 5, 120, 60)
        else:
            max_time = float('inf')
        if 'Cuisine' in df.columns:
            cuisine = st.multiselect("Cuisine", ['All'] + list(df['Cuisine'].unique()), default=['All'])
        else:
            cuisine = ['All']
        servings = st.number_input("Number of Servings", min_value=1, max_value=10, value=2)

    # Input section
    with st.container():
        st.subheader("Input Ingredients")
        input_method = st.radio("Choose input method:", ("Text Input", "Image Upload"))

        ingredients_list = []
        if input_method == "Text Input":
            ingredients_input = st.text_area("Enter ingredients (comma-separated, e.g., eggs, bread, cheese):", height=100)
            ingredients_list = [ing.strip() for ing in ingredients_input.split(',')] if ingredients_input else []
        else:
            uploaded_file = st.file_uploader("Upload an image of ingredients", type=['jpg', 'png', 'jpeg'])
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                ingredients = classify_image(uploaded_file, df, model)
                if ingredients:
                    ingredients_list = eval(ingredients) if isinstance(ingredients, str) else ingredients
                    st.write(f"**Detected Ingredients**: {', '.join(ingredients_list)}")

        if st.button("Find Recipes", key="find_recipes"):
            if not ingredients_list:
                st.warning("Please enter ingredients or upload an image!")
                return

            # Preprocess input
            processed_input = preprocess_ingredients(ingredients_list)

            # Vectorize ingredients
            vectorizer = TfidfVectorizer()
            recipe_vectors = vectorizer.fit_transform(df['Cleaned_Ingredients'])
            input_vector = vectorizer.transform([processed_input])

            # Calculate cosine similarity
            similarities = cosine_similarity(input_vector, recipe_vectors).flatten()
            df['Similarity'] = similarities

            # Filter recipes
            filtered_df = df.copy()
            if vegetarian:
                non_veg_ingredients = [
                    'chicken', 'beef', 'pork', 'fish', 'shrimp', 'lamb', 'turkey', 'duck', 'salmon',
                    'tuna', 'cod', 'bacon', 'sausage', 'ham', 'meat', 'crab', 'lobster', 'mutton'
                ]
                filtered_df = filtered_df[~filtered_df['Ingredients'].apply(
                    lambda x: any(ing.lower() in non_veg_ingredients for ing in (eval(x) if isinstance(x, str) else x))
                )]
            if 'All' not in cuisine and 'Cuisine' in df.columns:
                filtered_df = filtered_df[filtered_df['Cuisine'].isin(cuisine)]
            if 'Prep_Time' in df.columns:
                filtered_df = filtered_df[filtered_df['Prep_Time'] <= max_time]

            # Sort recipes
            if sort_by == "Prep Time" and 'Prep_Time' in df.columns:
                filtered_df = filtered_df.sort_values('Prep_Time')
            elif sort_by == "Calories" and 'Calories' in df.columns:
                filtered_df = filtered_df.sort_values('Calories')
            else:
                filtered_df = filtered_df.sort_values('Similarity', ascending=False)

            # Display results
            if filtered_df.empty:
                st.warning("No recipes found matching your criteria!")
            else:
                st.subheader("Matching Recipes")
                for idx, row in filtered_df.head(5).iterrows():
                    with st.container():
                        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                        st.write(f"### {row['Title']} (Similarity: {row['Similarity']:.2f})")
                        ingredients = eval(row['Ingredients']) if isinstance(row['Ingredients'], str) else row['Ingredients']
                        scaled_ingredients = scale_ingredients(ingredients, original_servings=2, target_servings=servings)
                        st.write(f"**Ingredients (for {servings} servings)**: {', '.join(scaled_ingredients)}")
                        
                        # Suggest substitutes for missing ingredients
                        recipe_ings = set(ing.lower() for ing in ingredients)
                        input_ings = set(ing.lower() for ing in ingredients_list)
                        missing_ings = recipe_ings - input_ings
                        if missing_ings:
                            st.write("**Missing Ingredients & Substitutes**:")
                            substitutes = suggest_substitutes(missing_ings)
                            for ing, subs in substitutes.items():
                                st.write(f"- {ing}: {', '.join(subs)}")

                        st.write(f"**Instructions**: {row['Instructions']}")
                        if 'Prep_Time' in df.columns:
                            st.write(f"**Prep Time**: {row['Prep_Time']} minutes")
                        if 'Calories' in df.columns:
                            st.write(f"**Calories**: {row['Calories']} kcal")
                        if 'Cuisine' in df.columns:
                            st.write(f"**Cuisine**: {row['Cuisine']}")
                        # Display image
                        image_name = row['Image_Name']
                        possible_extensions = ['', '.jpg', '.png', '.jpeg']
                        image_found = False
                        for ext in possible_extensions:
                            image_path = os.path.join("Food Images", image_name + ext)
                            if os.path.exists(image_path):
                                st.image(image_path, caption=row['Title'], use_container_width=True)
                                image_found = True
                                break
                        if not image_found:
                            st.write(f"(Image not found for {image_name})")
                        # Export recipe button
                        recipe_text = f"{row['Title']}\n\nIngredients (for {servings} servings):\n{', '.join(scaled_ingredients)}\n\nInstructions:\n{row['Instructions']}"
                        if 'Prep_Time' in df.columns:
                            recipe_text += f"\n\nPrep Time: {row['Prep_Time']} minutes"
                        if 'Calories' in df.columns:
                            recipe_text += f"\n\nCalories: {row['Calories']} kcal"
                        if 'Cuisine' in df.columns:
                            recipe_text += f"\n\nCuisine: {row['Cuisine']}"
                        st.download_button(
                            label="Download Recipe",
                            data=recipe_text,
                            file_name=f"{row['Title'].replace(' ', '_')}.txt",
                            mime="text/plain"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.write("---")

if __name__ == "__main__":
    main()
