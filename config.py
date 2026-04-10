import os
from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Image parameters
IMG_SIZE = (224, 224)  # EfficientNet-B0 input size
BATCH_SIZE = 32
NUM_CLASSES = 3  # normal, hypothyroidism, nodules

# Training parameters
EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 10  # for early stopping

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest',
    'brightness_range': [0.8, 1.2]
}

# Questionnaire parameters
QUESTIONNAIRE_WEIGHTS = {
    'hypothyroidism': {
        'fatigue': 0.2,
        'cold_sensitivity': 0.15,
        'weight_gain': 0.15,
        'dry_skin': 0.1,
        'constipation': 0.1,
        'depression': 0.1,
        'slow_heart_rate': 0.1,
        'muscle_weakness': 0.1
    },
    'nodules': {
        'neck_swelling': 0.3,
        'difficulty_swallowing': 0.2,
        'hoarseness': 0.15,
        'pain': 0.15,
        'breathing_difficulty': 0.2
    }
}

# Risk score thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 0.8
}

# Class names
CLASS_NAMES = ['normal', 'hypothyroidism', 'nodules'] 