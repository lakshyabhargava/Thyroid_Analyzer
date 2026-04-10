import tensorflow as tf
import numpy as np
from pathlib import Path
import config
from questionnaire import ThyroidQuestionnaire
from hormone_model import HormoneBasedModel
import data_preprocessing
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('thyroid_screening.log'),
        logging.StreamHandler()
    ]
)

class ThyroidScreeningSystem:
    def __init__(self, model_path="./models/best_model.h5", hormone_model_path='./models/best_thyroid_model_pipeline.pkl'):
        """
        Initialize the screening system.
        
        Args:
            model_path (str): Path to the trained image model file
            hormone_model_path (str): Path to the trained hormone model file
        """
        logging.info("Initializing ThyroidScreeningSystem")
        
        
        # Load the image model
        try:
            self.model = tf.keras.models.load_model(str(model_path))
            logging.info("Image model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load image model: {str(e)}")
            raise
        
        # Initialize questionnaire
        self.questionnaire = ThyroidQuestionnaire()
        logging.info("Questionnaire initialized")
        
        # Initialize hormone model
        logging.info(f"Initializing hormone model from: {hormone_model_path}")
        self.hormone_model = HormoneBasedModel(model_path=hormone_model_path)
        
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image
        """
        logging.info(f"Preprocessing image: {image_path}")
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (224, 224))  # Standard size for most pre-trained models
            img = tf.cast(img, tf.float32) / 255.0
            logging.info("Image preprocessing completed successfully")
            return np.expand_dims(img, axis=0)
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict_image(self, image_path):
        """
        Predict thyroid condition from image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Prediction results
        """
        logging.info(f"Making prediction for image: {image_path}")
        try:
            # Preprocess image
            img = self.preprocess_image(image_path)
            
            # Get prediction
            pred_probs = self.model.predict(img)[0]
            pred_class = np.argmax(pred_probs)
            
            result = {
                'class': config.CLASS_NAMES[pred_class],
                'confidence': float(pred_probs[pred_class]),
                'probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(config.CLASS_NAMES, pred_probs)
                }
            }
            logging.info(f"Image prediction completed. Predicted class: {result['class']} with confidence: {result['confidence']:.2f}")
            return result
        except Exception as e:
            logging.error(f"Error during image prediction: {str(e)}")
            raise
    
    def should_use_second_stage(self, image_pred, questionnaire_results):
        """
        Determine if second-stage model should be used.
        
        Args:
            image_pred (dict): Image prediction results
            questionnaire_results (dict): Questionnaire results
            
        Returns:
            bool: Whether to use second-stage model
        """
        # Get image confidence and questionnaire score
        image_confidence = image_pred['confidence']
        questionnaire_score = questionnaire_results['risk_score'] / 100  # Convert to 0-1 scale
        
        # Calculate combined score
        combined_score = (image_confidence * 0.85) + (questionnaire_score * 0.15)
        
        # Use second stage if combined score is between 25% and 39%
        return 0.25 <= combined_score < 0.39
    
    def combine_predictions(self, image_pred, questionnaire_results, hormone_results=None):
        """
        Combine predictions from different sources.
        
        Args:
            image_pred (dict): Image prediction results
            questionnaire_results (dict): Questionnaire results
            hormone_results (dict, optional): Hormone model results
            
        Returns:
            dict: Combined results
        """
        logging.info("Combining predictions")
        try:
            # Get image-based condition and confidence
            image_condition = image_pred['class']
            image_confidence = image_pred['confidence']
            
            # Get questionnaire-based condition and score
            questionnaire_condition = questionnaire_results['primary_condition']
            questionnaire_score = questionnaire_results['risk_score'] / 100  # Convert to 0-1 scale
            
            if hormone_results:
                # Use hormone model results if available
                final_condition = hormone_results['condition']
                combined_score = hormone_results['confidence']
                risk_level = hormone_results['risk_level']
                recommendation = hormone_results['recommendation']
                
                logging.info("Using hormone model results")
            else:
                # Calculate combined score
                combined_score = (image_confidence * 0.85) + (questionnaire_score * 0.15)
                
                # Determine final condition
                if image_confidence > 0.7:
                    final_condition = image_condition
                    logging.info(f"Using image-based condition due to high confidence: {final_condition}")
                else:
                    final_condition = questionnaire_condition
                    logging.info(f"Using questionnaire-based condition: {final_condition}")
                
                # Get risk level and recommendation
                risk_level = self.questionnaire.get_risk_level(combined_score)
                recommendation = self.questionnaire.get_recommendation(risk_level, final_condition)
            
            # Check if hormone analysis is needed
            needs_hormone_analysis = self.should_use_second_stage(image_pred, questionnaire_results)
            
            result = {
                'final_condition': final_condition,
                'combined_risk_score': combined_score * 100,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'image_prediction': image_pred,
                'questionnaire_results': questionnaire_results,
                'hormone_results': hormone_results,
                'needs_hormone_analysis': needs_hormone_analysis
            }
            
            logging.info(f"Combined prediction completed. Final condition: {final_condition}, Risk level: {risk_level}")
            return result
        except Exception as e:
            logging.error(f"Error combining predictions: {str(e)}")
            raise
    
    def screen_patient(self, image_path, questionnaire_responses, hormone_values=None):
        """
        Perform complete thyroid screening.
        
        Args:
            image_path (str): Path to the neck image
            questionnaire_responses (dict): Questionnaire responses
            hormone_values (dict, optional): Hormone test values
            
        Returns:
            dict: Complete screening results
        """
        logging.info("Starting patient screening process")
        try:
            # Get image prediction
            logging.info("Processing image prediction")
            image_pred = self.predict_image(image_path)
            
            # Process questionnaire
            logging.info("Processing questionnaire responses")
            questionnaire_results = self.questionnaire.process_responses(questionnaire_responses)
            
            # Check if second-stage model should be used
            if self.should_use_second_stage(image_pred, questionnaire_results) and hormone_values:
                logging.info("Using second-stage hormone model")
                hormone_results = self.hormone_model.predict(hormone_values, questionnaire_responses)
            else:
                hormone_results = None
            
            # Combine predictions
            logging.info("Combining predictions")
            results = self.combine_predictions(image_pred, questionnaire_results, hormone_results)
            
            logging.info("Patient screening completed successfully")
            return results
        except Exception as e:
            logging.error(f"Error during patient screening: {str(e)}")
            raise

# Example usage:
if __name__ == "__main__":
    logging.info("Starting thyroid screening system")
    try:
        # Initialize screening system
        model_path = "./models/best_thyroid_model_pipeline.pkl"
        screening_system = ThyroidScreeningSystem(hormone_model_path=model_path)
        
        # Example image path (replace with actual path)
        image_path = "path/to/neck_image.jpg"
        
        # Example questionnaire responses
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
        logging.info("Starting example screening")
        results = screening_system.screen_patient(image_path, responses)
        
        # Print results
        logging.info("Screening results:")
        print("\nThyroid Screening Results:")
        print(f"Final Condition: {results['final_condition']}")
        print(f"Combined Risk Score: {results['combined_risk_score']:.1f}%")
        print(f"Risk Level: {results['risk_level']}")
        print(f"Recommendation: {results['recommendation']}")
        
        print("\nImage Prediction Details:")
        print(f"Predicted Class: {results['image_prediction']['class']}")
        print(f"Confidence: {results['image_prediction']['confidence']:.2f}")
        
        print("\nQuestionnaire Results:")
        print(f"Primary Condition: {results['questionnaire_results']['primary_condition']}")
        print(f"Risk Score: {results['questionnaire_results']['risk_score']:.1f}%")
        
        logging.info("Example screening completed successfully")
    except Exception as e:
        logging.error(f"Error in example screening: {str(e)}")
        raise
