import os
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from pathlib import Path
import config
from predict import ThyroidScreeningSystem

app = Flask(__name__)

# Ensure model, data and results directories exist
for dir_path in [config.MODEL_DIR, config.DATA_DIR, config.RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Define the questionnaire items
QUESTIONNAIRE_ITEMS = {
    'Hypothyroidism Symptoms': [
        {'id': 'fatigue', 'text': 'Do you experience unusual fatigue or tiredness?'},
        {'id': 'cold_sensitivity', 'text': 'Do you have increased sensitivity to cold temperatures?'},
        {'id': 'weight_gain', 'text': 'Have you experienced unexplained weight gain?'},
        {'id': 'dry_skin', 'text': 'Do you have dry or rough skin?'},
        {'id': 'constipation', 'text': 'Do you experience constipation?'},
        {'id': 'depression', 'text': 'Have you been feeling unusually depressed?'},
        {'id': 'slow_heart_rate', 'text': 'Has anyone told you that you have a slow heart rate?'},
        {'id': 'muscle_weakness', 'text': 'Do you experience muscle weakness?'}
    ],
    'Nodule Symptoms': [
        {'id': 'neck_swelling', 'text': 'Have you noticed any swelling in your neck?'},
        {'id': 'difficulty_swallowing', 'text': 'Do you have difficulty swallowing?'},
        {'id': 'hoarseness', 'text': 'Have you experienced hoarseness in your voice?'},
        {'id': 'pain', 'text': 'Do you have pain in your neck or throat?'},
        {'id': 'breathing_difficulty', 'text': 'Do you experience any difficulty breathing?'}
    ]
}

# Initialize the screening system
screening_system = None

def initialize_screening_system():
    global screening_system
    
    # Look for the best available model
    if Path(config.MODEL_DIR / 'final_model.h5').exists():
        model_path = config.MODEL_DIR / 'final_model.h5'
    elif Path(config.MODEL_DIR / 'best_model.h5').exists():
        model_path = config.MODEL_DIR / 'best_model.h5'
    else:
        raise FileNotFoundError("No trained model found. Please train the model first.")
    
    screening_system = ThyroidScreeningSystem(model_path=str(model_path))

@app.route('/')
def index():
    global screening_system
    if screening_system is None:
        initialize_screening_system()
    return render_template('index.html', questionnaire=QUESTIONNAIRE_ITEMS)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})
    
    # Get the uploaded image
    image = request.files['image']
    
    # Check if a file was selected
    if image.filename == '':
        return jsonify({'error': 'No image selected'})
    
    # Save the image temporarily
    temp_path = config.DATA_DIR / 'temp_upload.jpg'
    image.save(str(temp_path))
    
    # Get questionnaire responses
    responses = {}
    for section in QUESTIONNAIRE_ITEMS.values():
        for item in section:
            item_id = item['id']
            # Check if the checkbox is checked (present in form data)
            responses[item_id] = item_id in request.form
    
    # Check for hormone values
    hormone_values = None
    if 'tsh' in request.form and 't3' in request.form and 't4' in request.form:
        hormone_values = {
            'TSH': float(request.form['tsh']),
            'T3': float(request.form['t3']),
            'T4': float(request.form['t4'])
        }
    
    # Perform screening
    results = screening_system.screen_patient(str(temp_path), responses, hormone_values)
    
    # Read image for display
    with open(temp_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Format results for display
    formatted_results = {
        'final_condition': results['final_condition'],
        'risk_score': f"{results['combined_risk_score']:.1f}%",
        'risk_level': results['risk_level'],
        'recommendation': results['recommendation'],
        'image_prediction': results['image_prediction']['class'],
        'image_confidence': f"{results['image_prediction']['confidence'] * 100:.1f}%",
        'questionnaire_result': results['questionnaire_results']['primary_condition'],
        'questionnaire_score': f"{results['questionnaire_results']['risk_score']:.1f}%",
        'image_data': img_data,
        'needs_hormone_analysis': results.get('needs_hormone_analysis', False)
    }
    
    # Add hormone results if available
    if results['hormone_results']:
        formatted_results['hormone_prediction'] = results['hormone_results']['condition']
        formatted_results['hormone_confidence'] = f"{results['hormone_results']['confidence'] * 100:.1f}%"
    
    # Clean up temp file
    os.remove(temp_path)
    
    return jsonify(formatted_results)

if __name__ == '__main__':
    app.run(debug=True) 