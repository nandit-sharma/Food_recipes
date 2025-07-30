# Recipe Recommender

A Streamlit-based recipe recommendation system that suggests recipes based on available ingredients.

## Features

- **Recipe Search**: Enter ingredients and get personalized recipe recommendations
- **Image Display**: Shows recipe images from local Food Images directory with intelligent fallback
- **Dietary Filters**: Filter recipes by vegetarian/non-vegetarian preferences
- **Bookmarking**: Save your favorite recipes for later
- **Smart Matching**: Uses TF-IDF vectorization and cosine similarity for accurate recommendations
- **Find Recipe Button**: Prominent button to trigger recipe search
- **Fast Loading**: Optimized for quick startup with lazy loading and caching

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the following files in your project directory:
   - `food_dataset.csv` - Recipe dataset
   - `Food Images/` - Directory containing recipe images
   - `app.py` - Main Streamlit application

## Usage

### Quick Start (Recommended)
For fastest startup, use one of these methods:

**Windows (Batch):**
```bash
run_app.bat
```

**Windows (PowerShell):**
```powershell
.\run_app.ps1
```

**Manual:**
```bash
python startup.py
streamlit run app.py
```

### Standard Method
```bash
streamlit run app.py
```

### Using the App

1. Open your browser and navigate to the provided URL (usually http://localhost:8501)

2. Enter ingredients separated by commas (e.g., "tomato, onion, garlic")

3. Click "Find Recipes" to get recommendations

4. Use the sidebar filters to customize your search

5. Bookmark recipes you like for future reference

## Performance Optimizations

The app has been optimized for faster loading:

- **Lazy Loading**: Data is only loaded when you first search for recipes
- **Caching**: TF-IDF vectors are cached for 1 hour to avoid recomputation
- **Progress Indicators**: Visual feedback during data loading and processing
- **Reduced Features**: Limited TF-IDF features to 5000 for faster processing
- **Session State**: Data persists between searches within the same session
- **Preloading Script**: Optional startup script to preload data before launching the app

## Project Structure

```
recipe_recommender/
├── app.py                 # Main Streamlit application
├── startup.py             # Data preloading script
├── run_app.bat           # Windows batch file for quick start
├── run_app.ps1           # PowerShell script for quick start
├── food_dataset.csv       # Recipe dataset
├── Food Images/          # Recipe images directory
├── requirements.txt       # Python dependencies
├── test_app.py           # Image loading test script
└── README.md            # This file
```

## How It Works

1. **Data Loading**: Loads recipe data from CSV file and matches images from local directory
2. **Text Processing**: Cleans and vectorizes ingredient text using TF-IDF
3. **Similarity Matching**: Uses cosine similarity to find recipes matching user ingredients
4. **Image Matching**: Fuzzy matches recipe names to image filenames using sequence matching
5. **Filtering**: Applies dietary and preference filters
6. **Display**: Shows recipes with images, ingredients, and instructions
7. **Fallback System**: Creates local placeholder images when actual images are missing

## Technical Details

- **Framework**: Streamlit
- **ML Library**: scikit-learn (TF-IDF, cosine similarity)
- **Image Processing**: PIL (Python Imaging Library) with local fallback system
- **Text Processing**: NLTK (Natural Language Toolkit)
- **Data Handling**: pandas, numpy
- **Image Matching**: Fuzzy string matching using difflib.SequenceMatcher
- **Caching**: Streamlit cache_data with TTL for performance
- **Session Management**: Persistent session state for data reuse

## Recent Improvements

- ✅ **Performance Optimization**: Lazy loading and caching for faster startup
- ✅ **Progress Indicators**: Visual feedback during data processing
- ✅ **Startup Scripts**: Batch and PowerShell scripts for quick launch
- ✅ **Reduced Memory Usage**: Limited TF-IDF features and optimized data structures
- ✅ **Better Error Handling**: Graceful fallback for missing images and data
- ✅ **Enhanced User Experience**: Faster response times and better visual feedback

## Troubleshooting

- If images don't display, ensure the `Food Images/` directory exists and contains `.jpg` files
- If recipes don't load, check that `food_dataset.csv` is in the correct location
- For dependency issues, run `pip install -r requirements.txt`
- The app now uses only local resources - no external image URLs
- If the app is slow to start, use the startup scripts to preload data

## Performance Tips

- First search may take 10-15 seconds to load and process data
- Subsequent searches will be much faster due to caching
- Use the startup scripts for the fastest initial loading experience
- The app caches data for 1 hour, so restarting within that time will be faster

## License

This project is for educational purposes. 