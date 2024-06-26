import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import pickle
import instaloader

# Load the dataset
data = pd.read_csv("C:/Users/MOHITH/Desktop/collage/ML/mbti_1.csv")

# Basic data exploration
print(data.posts[0])
print(data.describe())
print(data['type'].value_counts())

# Visualizations
ax = pd.DataFrame(data.type.value_counts()).plot.bar(color='lightgreen')
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', color='brown')

plt.ylabel('Number posts available for each Personality')
plt.xlabel('Types of Personalities')
plt.title('Bar graph showing frequency of different types of personalities')
plt.show()

# Stratified split to ensure equal distribution of data
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data.type)

# Text cleaning function
def clear_text(data):
    data_length = []
    lemmatizer = WordNetLemmatizer()
    remove_words = stopwords.words("english")
    cleaned_text = []
    for sentence in tqdm(data.posts):
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

# Clean the text data
train_data.posts, train_length = clear_text(train_data)
test_data.posts, test_length = clear_text(test_data)

# Custom lemmatizer class
class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', tokenizer=Lemmatizer())
vectorizer.fit(train_data.posts)

train_post = vectorizer.transform(train_data.posts).toarray()
test_post = vectorizer.transform(test_data.posts).toarray()

# Encode the target labels
target_encoder = LabelEncoder()
train_target = target_encoder.fit_transform(train_data.type)
test_target = target_encoder.fit_transform(test_data.type)

# Train Linear Support Vector Classifier
model_linear_svc = LinearSVC(C=0.1)
model_linear_svc.fit(train_post, train_target)

# Print classification report
print('Train classification report\n', classification_report(train_target, model_linear_svc.predict(train_post), target_names=target_encoder.inverse_transform([i for i in range(16)])))
print('Test classification report\n', classification_report(test_target, model_linear_svc.predict(test_post), target_names=target_encoder.inverse_transform([i for i in range(16)])))

# Save the model and vectorizer to pickle files
with open('linear_svc_model.pkl', 'wb') as model_file:
    pickle.dump(model_linear_svc, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(target_encoder, encoder_file)

print("Model, vectorizer, and label encoder have been saved to pickle files.")

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
    username = "arshgoyalyt"
    cls = GetInstagramProfile()
    user_df = cls.get_post_info_csv(username)
    print(user_df.posts[0])

    user_df_length = len(user_df)
    user_df.posts, user_df_length = clear_text(user_df)
    print(user_df.posts[0])

    # Load the saved model, vectorizer, and label encoder
    with open('linear_svc_model.pkl', 'rb') as model_file:
        model_linear_svc = pickle.load(model_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    with open('label_encoder.pkl', 'rb') as encoder_file:
        target_encoder = pickle.load(encoder_file)

    user_post = vectorizer.transform(user_df.posts).toarray()
    # Make predictions for the new user
    user_pred = model_linear_svc.predict(user_post)

    # Decode the predicted personality type
    user_personality = target_encoder.inverse_transform(user_pred)

    # Print the predicted personality for the new user
    print("Predicted personality for the new user:", user_personality[0])
