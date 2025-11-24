
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import requests

app = Flask(__name__)

# Load Ohm's Law model
with open('models/ohms_law_model.pkl', 'rb') as f:
    ohms_model = pickle.load(f)

API_KEY = "YOUR_API_KEY_HERE"  # Replace with your GPT API key

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
