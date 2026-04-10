import os
import logging
from pathlib import Path
from app import app
import config

# Create required directories if they don't exist
os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":
    print("Starting Thyroid Disease Screening System...")
    print(f"Model directory: {config.MODEL_DIR}")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Visit http://127.0.0.1:5000/ in your browser to use the application")
    
    # Verify model files
    model_files = list(Path(config.MODEL_DIR).glob('*.h5'))
    if model_files:
        print(f"Found model files: {', '.join([f.name for f in model_files])}")
    else:
        print("Warning: No model files found in models directory. Please add a model file.")
    
    # Check templates directory
    if not os.path.exists('templates/index.html'):
        print("Warning: templates/index.html not found. Application may not work correctly.")
    
    # Run the Flask application
    app.run(debug=True, host='127.0.0.1', port=5000) 