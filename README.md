# Thyroid Disease Screening System

A machine learning-based early screening system for thyroid diseases, designed for use in rural or low-resource settings. This is a pilot implementation that combines image analysis and symptom-based questionnaire responses to assess thyroid health.

## Features

- Image-based classification using EfficientNet-B0
- Symptom-based questionnaire scoring
- Combined risk assessment
- Clear recommendations for next steps
- Designed for low-resource settings

## System Requirements

- Python 3.8+
- TensorFlow 2.12+
- See `requirements.txt` for full dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd thyroid-screening
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
thyroid-screening/
├── config.py              # Configuration parameters
├── data_preprocessing.py  # Image preprocessing and augmentation
├── model.py              # Model architecture and training
├── questionnaire.py      # Symptom scoring logic
├── predict.py            # Combined prediction system
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Usage

### Training the Model

1. Place your training images in the appropriate directories:
   - `data/normal/`
   - `data/hypothyroidism/`
   - `data/hyperthyroidism/`
   - `data/nodules/`

2. Run the training script:
```bash
python model.py
```

### Making Predictions

1. Initialize the screening system:
```python
from predict import ThyroidScreeningSystem
screening_system = ThyroidScreeningSystem()
```

2. Perform screening:
```python
results = screening_system.screen_patient(
    image_path="path/to/neck_image.jpg",
    questionnaire_responses={
        'fatigue': True,
        'cold_sensitivity': True,
        # ... other symptoms
    }
)
```

## Output

The system provides:
- Thyroid Risk Score (0-100%)
- Primary predicted condition
- Risk level (low/medium/high)
- Clear recommendation for next steps
- Detailed breakdown of image and questionnaire results

## Limitations

- This is a pilot implementation, not a clinical product
- Limited to the four main thyroid conditions
- Requires clear frontal neck images
- Accuracy may vary based on image quality and lighting

## Future Improvements

- Integration with ultrasound data
- Mobile app implementation
- Multi-language support
- Additional condition detection
- Improved model accuracy with more training data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- EfficientNet paper and implementation
- Medical professionals who provided guidance
- Open-source community contributions 