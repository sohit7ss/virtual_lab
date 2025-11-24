
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import requests
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Ensure model exists or create dynamically
MODEL_PATH = 'models/ohms_law_model.pkl'
if not os.path.exists(MODEL_PATH):
    os.makedirs('models', exist_ok=True)
    voltages = np.linspace(1, 100, 100)
    resistances = np.linspace(1, 50, 100)
    X, y = [], []
    for v in voltages:
        for r in resistances:
            X.append([v, r])
            y.append(v / r)
    model = LinearRegression()
    model.fit(X, y)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

# Load the model
with open(MODEL_PATH, 'rb') as f:
    ohms_model = pickle.load(f)

# API Key for GPT
API_KEY = os.getenv("API_KEY")  # Use environment variable for security

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/experiment')
def experiment():
    return render_template('experiment.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    voltage = float(data.get('voltage'))
    resistance = float(data.get('resistance'))
    prediction = ohms_model.predict([[voltage, resistance]])[0]
    return jsonify({'predicted_current': prediction})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": user_message}]
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True)
