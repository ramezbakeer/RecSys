from flask import request, jsonify
import numpy as np
from app import app
from app.preprocess import preprocess_text, vectorize_text, connect_to_database, get_all_job_vectors, get_all_problem_vectors, get_user_vector
from sklearn.metrics.pairwise import cosine_similarity
import json

@app.route('/')
def home():
    return "Flask app is running!"

# Function to add CORS headers
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Endpoint to recommend jobs and problems to a user
@app.route('/recommendations', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data.get('user_id', '')
    if user_id:
        conn = connect_to_database()
        user_vector = get_user_vector(user_id, conn)
        
        if user_vector is None:
            return jsonify({'error': 'User vector not found.'}), 404
        
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

# Endpoint to vectorize user bio and profession
@app.route('/vectorize_user', methods=['POST'])
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

# Endpoint to vectorize job title and description
@app.route('/vectorize_job', methods=['POST'])
def vectorize_job():
    data = request.get_json()
    job_id = data.get('job_id', '')
    title = data.get('title', '')
    description = data.get('description', '')
    if job_id and title and description:
        combined_text = title + " " + description
        job_vector = vectorize_text(combined_text)
        return jsonify({'job_id': job_id, 'vector': job_vector.tolist()})
    else:
        return jsonify({'error': 'Job ID, title, and description are required.'}), 400

# Endpoint to vectorize problem name and description
@app.route('/vectorize_problem', methods=['POST'])
def vectorize_problem():
    data = request.get_json()
    problem_id = data.get('problem_id', '')
    name = data.get('name', '')
    description = data.get('description', '')
    if problem_id and name and description:
        combined_text = name + " " + description
        problem_vector = vectorize_text(combined_text)
        return jsonify({'problem_id': problem_id, 'vector': problem_vector.tolist()})
    else:
        return jsonify({'error': 'Problem ID, name, and description are required.'}), 400
