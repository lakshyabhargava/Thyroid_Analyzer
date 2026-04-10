import os
import base64
from flask import Flask, render_template_string, request, jsonify
from pathlib import Path
import config
from predict import ThyroidScreeningSystem

app = Flask(__name__)

# --- 1. EMBEDDED HTML INTERFACE (No separate file needed) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Thyroid Disease Screening AI</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #f0f2f5; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .section { margin-bottom: 25px; padding: 15px; background: #f8f9fa; border-radius: 8px; }
        .checkbox-group { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .btn { background-color: #3498db; color: white; padding: 12px 20px; border: none; border-radius: 5px; cursor: pointer; width: 100%; font-size: 16px; }
        .btn:hover { background-color: #2980b9; }
        #results { display: none; margin-top: 20px; border-top: 2px solid #eee; padding-top: 20px; }
        .result-card { background: #e8f6f3; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .risk-high { color: #c0392b; font-weight: bold; }
        .risk-low { color: #27ae60; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 Thyroid Screening System</h1>
        
        <form id="screeningForm">
            <div class="section">
                <h3>1. Upload Ultrasound Image</h3>
                <input type="file" name="image" accept=".jpg,.jpeg,.png" required>
            </div>

            <div class="section">
                <h3>2. Symptoms</h3>
                {% for category, items in questionnaire.items() %}
                    <h4>{{ category }}</h4>
                    <div class="checkbox-group">
                        {% for item in items %}
                            <label>
                                <input type="checkbox" name="{{ item.id }}"> {{ item.text }}
                            </label>
                        {% endfor %}
                    </div>
                {% endfor %}
            </div>

            <button type="button" class="btn" onclick="submitForm()">Analyze Patient</button>
        </form>

        <div id="results">
            <h2>Analysis Results</h2>
            <div id="loading" style="text-align: center; display: none;">Processing...</div>
            <div id="output"></div>
        </div>
    </div>

    <script>
        async function submitForm() {
            const form = document.getElementById('screeningForm');
            const formData = new FormData(form);
            const outputDiv = document.getElementById('output');
            const loadingDiv = document.getElementById('loading');
            
            document.getElementById('results').style.display = 'block';
            loadingDiv.style.display = 'block';
            outputDiv.innerHTML = '';

            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.error) {
                    outputDiv.innerHTML = `<p style="color:red">Error: ${data.error}</p>`;
                } else {
                    outputDiv.innerHTML = `
                        <div class="result-card">
                            <h3>Final Condition: ${data.final_condition}</h3>
                            <p>Risk Level: <span class="${data.risk_level === 'High' ? 'risk-high' : 'risk-low'}">${data.risk_level}</span></p>
                            <p>Combined Risk Score: ${data.risk_score}</p>
                            <p><strong>Recommendation:</strong> ${data.recommendation}</p>
                        </div>
                        <div style="display: flex; gap: 20px;">
                            <div style="flex: 1;">
                                <h4>Image Analysis</h4>
                                <p>Prediction: ${data.image_prediction}</p>
                                <p>Confidence: ${data.image_confidence}</p>
                                <img src="data:image/jpeg;base64,${data.image_data}" width="100%" style="border-radius: 5px;">
                            </div>
                            <div style="flex: 1;">
                                <h4>Symptom Analysis</h4>
                                <p>Primary: ${data.questionnaire_result}</p>
                                <p>Score: ${data.questionnaire_score}</p>
                            </div>
                        </div>
                    `;
                }
            } catch (error) {
                outputDiv.innerHTML = `<p style="color:red">System Error: ${error}</p>`;
            } finally {
                loadingDiv.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

# --- 2. PYTHON BACKEND LOGIC ---

# Ensure directories exist
for dir_path in [config.MODEL_DIR, config.DATA_DIR, config.RESULTS_DIR]:
    if hasattr(dir_path, 'mkdir'):
        dir_path.mkdir(exist_ok=True)

# Questionnaire Configuration
QUESTIONNAIRE_ITEMS = {
    'Hypothyroidism': [
        {'id': 'fatigue', 'text': 'Do you experience unusual fatigue?'},
        {'id': 'cold_sensitivity', 'text': 'Increased sensitivity to cold?'},
        {'id': 'weight_gain', 'text': 'Unexplained weight gain?'},
        {'id': 'dry_skin', 'text': 'Dry or rough skin?'},
        {'id': 'constipation', 'text': 'Constipation?'},
        {'id': 'depression', 'text': 'Feeling depressed?'},
        {'id': 'slow_heart_rate', 'text': 'Slow heart rate?'},
        {'id': 'muscle_weakness', 'text': 'Muscle weakness?'}
    ],
    'Nodules / Structural': [
        {'id': 'neck_swelling', 'text': 'Swelling in your neck?'},
        {'id': 'difficulty_swallowing', 'text': 'Difficulty swallowing?'},
        {'id': 'hoarseness', 'text': 'Hoarseness in voice?'},
        {'id': 'pain', 'text': 'Pain in neck/throat?'},
        {'id': 'breathing_difficulty', 'text': 'Difficulty breathing?'}
    ]
}

# Initialize System
screening_system = None

def initialize_screening_system():
    global screening_system
    if Path(config.MODEL_DIR / 'final_model.h5').exists():
        model_path = config.MODEL_DIR / 'final_model.h5'
    elif Path(config.MODEL_DIR / 'best_model.h5').exists():
        model_path = config.MODEL_DIR / 'best_model.h5'
    else:
        # Fallback for testing if no model exists yet
        print("WARNING: No model found. Please ensure 'final_model.h5' is in models folder.")
        return
    
    screening_system = ThyroidScreeningSystem(model_path=str(model_path))

# Initialize immediately
with app.app_context():
    initialize_screening_system()

@app.route('/')
def index():
    # Renders the string directly - no folder needed!
    return render_template_string(HTML_TEMPLATE, questionnaire=QUESTIONNAIRE_ITEMS)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'error': 'No image selected'})
    
    image = request.files['image']
    temp_path = config.DATA_DIR / 'temp_upload.jpg'
    image.save(str(temp_path))
    
    responses = {}
    for section in QUESTIONNAIRE_ITEMS.values():
        for item in section:
            responses[item['id']] = item['id'] in request.form
    
    # Logic to prevent crash if model isn't loaded
    if not screening_system:
        return jsonify({'error': 'Model not loaded. Check server logs.'})

    results = screening_system.screen_patient(str(temp_path), responses)
    
    with open(temp_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    formatted_results = {
        'final_condition': results['final_condition'],
        'risk_score': f"{results['combined_risk_score']:.1f}%",
        'risk_level': results['risk_level'],
        'recommendation': results['recommendation'],
        'image_prediction': results['image_prediction']['class'],
        'image_confidence': f"{results['image_prediction']['confidence'] * 100:.1f}%",
        'questionnaire_result': results['questionnaire_results']['primary_condition'],
        'questionnaire_score': f"{results['questionnaire_results']['risk_score']:.1f}%",
        'image_data': img_data
    }
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return jsonify(formatted_results)

if __name__ == '__main__':
    app.run(debug=True)
