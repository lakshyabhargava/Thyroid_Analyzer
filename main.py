import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Thyroid Disease Screening System")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Make predictions using the trained model')
    parser.add_argument('--image', type=str, help='Path to the image for prediction')
    
    args = parser.parse_args()
    
    if args.train:
        print("Training the model...")
        import model
        model.main()
    
    elif args.predict:
        if not args.image:
            print("Error: Please provide an image path with --image")
            return
            
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found")
            return
            
        print(f"Making predictions for image: {image_path}")
        
        from predict import ThyroidScreeningSystem
        import config
        
        # Check if model exists
        model_path = config.MODEL_DIR / 'final_model.h5'
        if not model_path.exists():
            model_path = config.MODEL_DIR / 'best_model.h5'
            if not model_path.exists():
                print("Error: No trained model found. Please train the model first with --train")
                return
        
        # Initialize the screening system
        screening_system = ThyroidScreeningSystem(model_path=str(model_path))
        
        # Example questionnaire responses (this would normally come from user input)
        responses = {
            'fatigue': True,
            'cold_sensitivity': True,
            'weight_gain': True,
            'dry_skin': False,
            'constipation': True,
            'depression': False,
            'slow_heart_rate': False,
            'muscle_weakness': True,
            'neck_swelling': True,
            'difficulty_swallowing': False,
            'hoarseness': False,
            'pain': False,
            'breathing_difficulty': False
        }
        
        # Perform screening
        results = screening_system.screen_patient(image_path, responses)
        
        # Print results
        print("\n===== Thyroid Screening Results =====")
        print(f"Final Condition: {results['final_condition']}")
        print(f"Combined Risk Score: {results['combined_risk_score']:.1f}%")
        print(f"Risk Level: {results['risk_level']}")
        print(f"Recommendation: {results['recommendation']}")
        
        print("\n=== Image Prediction Details ===")
        print(f"Predicted Class: {results['image_prediction']['class']}")
        print(f"Confidence: {results['image_prediction']['confidence']:.2f}")
        
        print("\n=== Class Probabilities ===")
        for class_name, prob in results['image_prediction']['probabilities'].items():
            print(f"{class_name}: {prob:.2f}")
        
        print("\n=== Questionnaire Results ===")
        print(f"Primary Condition: {results['questionnaire_results']['primary_condition']}")
        print(f"Risk Score: {results['questionnaire_results']['risk_score']:.1f}%")
        
        print("\n=== Symptom Scores ===")
        for condition, score in results['questionnaire_results']['symptom_scores'].items():
            print(f"{condition}: {score:.2f}")
    
    else:
        print("Please specify either --train or --predict")
        print("Example usage:")
        print("  Train the model:   python main.py --train")
        print("  Make prediction:   python main.py --predict --image path/to/image.jpg")

if __name__ == "__main__":
    main() 
