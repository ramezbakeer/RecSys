from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import mysql.connector
import numpy as np
import json

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(preprocessor=lambda x: x)

# Define preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]  # Remove non-alphabetic tokens
    tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stop words
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatization
    return ' '.join(tokens)

# Function to vectorize text
def vectorize_text(text):
    preprocessed_text = preprocess_text(text)
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])
    return tfidf_matrix.toarray()[0]

# Connect to MySQL database
def connect_to_database():
    return mysql.connector.connect(
        host='127.0.0.1',
        user='rash_rashahly',
        password='NlvoJ6%MnCjlbP5a',
        database='rash_rashahly'
    )

# Function to retrieve all job vectors from database
def get_all_job_vectors(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT job_id, tfidf_vector FROM job_tfidf_vectors")
    job_vectors = cursor.fetchall()
    cursor.close()
    return job_vectors

# Function to retrieve all problem vectors from database
def get_all_problem_vectors(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT problem_id, tfidf_vector FROM problem_tfidf_vectors")
    problem_vectors = cursor.fetchall()
    cursor.close()
    return problem_vectors

# Function to retrieve user vector from database
def get_user_vector(user_id, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT tfidf_vector FROM user_tfidf_vectors WHERE user_id = %s", (user_id,))
    result = cursor.fetchone()
    cursor.close()
    if result:
        return np.array(json.loads(result[0]))
    else:
        return None
