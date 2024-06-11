from flask import Blueprint, request, jsonify
import pymysql
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocess import preprocess_text
import json
import os

main = Blueprint('main', __name__)

def connect_to_database():
    return pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        port=int(os.getenv('DB_PORT')),
        connect_timeout=10  # You can adjust the timeout value as needed
    )
    
tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_text)

def vectorize_text(text):
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    return tfidf_matrix.toarray()[0]

def get_all_job_vectors(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT id, vector FROM jobs")
    job_vectors = cursor.fetchall()
    cursor.close()
    return job_vectors
#get 
def get_all_problem_vectors(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT id, vector FROM problems")
    problem_vectors = cursor.fetchall()
    cursor.close()
    return problem_vectors

def get_user_vector(user_id, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT vectorize_user FROM profile_types WHERE user_id = %s", (user_id,))
    result = cursor.fetchone()
    cursor.close()
    if result:
        return np.array(json.loads(result[0]))
    else:
        return None

@main.route('/recommendations', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data.get('user_id', '')
    if user_id:
        conn = connect_to_database()
        user_vector = get_user_vector(user_id, conn)
        job_vectors = get_all_job_vectors(conn)
        job_recommendations = []
        for job_id, job_vector in job_vectors:
            job_vector = np.array(json.loads(job_vector))
            similarity_score = cosine_similarity([user_vector], [job_vector])[0][0]
            job_recommendations.append((job_id, similarity_score))
        
        job_recommendations.sort(key=lambda x: x[1], reverse=True)
        top_job_recommendations = job_recommendations[:10]

        problem_vectors = get_all_problem_vectors(conn)
        problem_recommendations = []
        for problem_id, problem_vector in problem_vectors:
            problem_vector = np.array(json.loads(problem_vector))
            similarity_score = cosine_similarity([user_vector], [problem_vector])[0][0]
            problem_recommendations.append((problem_id, similarity_score))
        
        problem_recommendations.sort(key=lambda x: x[1], reverse=True)
        top_problem_recommendations = problem_recommendations[:10]

        conn.close()
        return jsonify({
            'user_id': user_id,
            'top_job_recommendations': top_job_recommendations,
            'top_problem_recommendations': top_problem_recommendations
        })
    else:
        return jsonify({'error': 'User ID is required.'}), 400

@main.route('/vectorize_user', methods=['POST'])
def vectorize_user():
    data = request.get_json()
    user_id = data.get('user_id', '')
    bio = data.get('bio', '')
    profession = data.get('profession', '')
    if user_id and bio and profession:
        combined_text = bio + " " + profession
        user_vector = vectorize_text(combined_text)
        return jsonify({'user_id': user_id, 'vector': user_vector.tolist()})
    else:
        return jsonify({'error': 'User ID, bio, and profession are required.'}), 400

@main.route('/vectorize_job', methods=['POST'])
def vectorize_job():
    data = request.get_json()
    job_id = data.get('id', '')
    title = data.get('name', '')
    description = data.get('description', '')
    if job_id and title and description:
        combined_text = title + " " + description
        job_vector = vectorize_text(combined_text)
        return jsonify({'id': job_id, 'vector': job_vector.tolist()})
    else:
        return jsonify({'error': 'ID, name, and description are required.'}), 400

@main.route('/vectorize_problem', methods=['POST'])
def vectorize_problem():
    data = request.get_json()
    problem_id = data.get('id', '')
    name = data.get('name', '')
    description = data.get('description', '')
    if problem_id and name and description:
        combined_text = name + " " + description
        problem_vector = vectorize_text(combined_text)
        return jsonify({'id': problem_id, 'vector': problem_vector.tolist()})
    else:
        return jsonify({'error': 'Problem ID, name, and description are required.'}), 400
