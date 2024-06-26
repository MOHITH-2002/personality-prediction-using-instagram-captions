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

# Text cleaning function
def clear_text(data):
    data_length = []
    lemmatizer = WordNetLemmatizer()
    remove_words = stopwords.words("english")
    cleaned_text = []
    for sentence in data:
        sentence = sentence.lower()
        sentence = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', sentence)
        sentence = re.sub('[^0-9a-z]', ' ', sentence)
        sentence = re.sub('\s+', ' ', sentence)
        sentence = re.sub('\W+', ' ', sentence)
        sentence = re.sub('[0-9]', ' ', sentence)
        sentence = re.sub('[_+]', ' ', sentence)
        sentence = " ".join([w for w in sentence.split() if w not in remove_words])
        sentence = " ".join([lemmatizer.lemmatize(w) for w in sentence.split()])
        data_length.append(len(sentence.split()))
        cleaned_text.append(sentence)
    return cleaned_text, data_length

# Load the saved model, vectorizer, and label encoder
def load_model_vectorizer_encoder(model_path, vectorizer_path, encoder_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    with open(encoder_path, 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)

    return model, vectorizer, encoder

# Instagram caption extraction
class GetInstagramProfile():
    def __init__(self):
        self.L = instaloader.Instaloader()

    def get_post_info_csv(self, username):
        captions = []  # List to store captions
        posts = instaloader.Profile.from_username(self.L.context, username).get_posts()

        for post in posts:
            caption = post.caption if post.caption is not None else " "
            captions.append(caption)

        # Create a DataFrame with a single row and a column containing all captions
        user_df = pd.DataFrame({'posts': ['\n'.join(captions)]})
        return user_df

if __name__ == "__main__":
    # Check if username argument is provided
    if len(sys.argv) != 2:
        print("Usage: python predict_instagram_user.py <username>")
        sys.exit(1)
    
    # Extract username from command-line argument
    username = sys.argv[1]

    # Initialize Instagram profile retrieval class
    cls = GetInstagramProfile()

    # Get user's Instagram captions
    user_df = cls.get_post_info_csv(username)

    # Clean the text data
    user_posts, _ = clear_text(user_df['posts'])

    # Paths to saved model, vectorizer, and encoder
    model_path = 'linear_svc_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    encoder_path = 'label_encoder.pkl'

    # Load model, vectorizer, and encoder
    model, vectorizer, encoder = load_model_vectorizer_encoder(model_path, vectorizer_path, encoder_path)

    # Vectorize user posts
    user_post = vectorizer.transform(user_posts).toarray()

    # Predict user personality
    user_pred = model.predict(user_post)

    # Decode predicted personality
    user_personality = encoder.inverse_transform(user_pred)

    # Print predicted personality
    print(f"Predicted personality for {username}: {user_personality[0]}")
