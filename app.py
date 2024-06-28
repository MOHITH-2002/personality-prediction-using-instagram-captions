from flask import Flask, render_template, request, jsonify
import sys
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import re
import instaloader
from text_processing import Lemmatizer  # Import the Lemmatizer class

# Initialize NLTK components
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK components are initialized
nltk.download('wordnet')
nltk.download('stopwords')

# Import functions from model.py
from model import load_model_vectorizer_encoder, GetInstagramProfile, clear_text

app = Flask(__name__)

# # Load model, vectorizer, and encoder
# model_path = 'linear_svc_model.pkl'
# vectorizer_path = 'tfidf_vectorizer.pkl'
# encoder_path = 'label_encoder.pkl'
# model, vectorizer, encoder = load_model_vectorizer_encoder(model_path, vectorizer_path, encoder_path)

# # Initialize Instagram profile retrieval class
# cls = GetInstagramProfile()

# # Text cleaning function
# def clean_user_posts(username):
#     # Get user's Instagram captions
#     user_df = cls.get_post_info_csv(username)

#     # Clean the text data
#     user_posts, _ = clear_text(user_df['posts'])

#     return user_posts

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/', methods=['POST'])
# def predict():
#     username = request.form['username']

#     if not username:
#         return jsonify({'error': 'Username not provided.'}), 400

#     try:
#         # Clean user posts
#         user_posts = clean_user_posts(username)

#         # Vectorize user posts
#         user_post = vectorizer.transform(user_posts).toarray()

#         # Predict user personality
#         user_pred = model.predict(user_post)

#         # Decode predicted personality
#         user_personality = encoder.inverse_transform(user_pred)

#         # Return predicted personality
#         return jsonify({'username': username, 'personality': user_personality[0]})
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


# app = Flask(__name__)

# # Initialize NLTK components
# nltk.download('wordnet')
# nltk.download('stopwords')

# Load model, vectorizer, and encoder
model_path = 'linear_svc_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'
encoder_path = 'label_encoder.pkl'
model, vectorizer, encoder = load_model_vectorizer_encoder(model_path, vectorizer_path, encoder_path)

# Initialize Instagram profile retrieval class
cls = GetInstagramProfile()

# Text cleaning function
def clean_user_posts(username):
    # Get user's Instagram captions
    user_df = cls.get_post_info_csv(username)

    # Clean the text data
    user_posts, _ = clear_text(user_df['posts'])

    return user_posts

@app.route('/')
def index():
    return 'Welcome to Personality Prediction from Instagram Posts!'

@app.route('/predict/<username>', methods=['GET'])
def predict(username):
    if not username:
        return jsonify({'error': 'Username not provided.'}), 400

    try:
        # Clean user posts
        user_posts = clean_user_posts(username)

        # Vectorize user posts
        user_post = vectorizer.transform(user_posts).toarray()

        # Predict user personality
        user_pred = model.predict(user_post)

        # Decode predicted personality
        user_personality = encoder.inverse_transform(user_pred)

        # Return predicted personality
        return jsonify({'username': username, 'personality': user_personality[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
