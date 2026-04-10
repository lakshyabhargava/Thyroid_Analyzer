import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('thyroid_screening.log'),
        logging.StreamHandler()
    ]
)

class HormoneBasedModel:
    def __init__(self, model_path=None):
        """
        Initialize and load the trained hormone-based model pipeline.
        """
        try:
            logging.info(f"Loading model from: {model_path}")
            self.pipeline = joblib.load(model_path)
            logging.info("Model pipeline loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

        self.symptom_weights = {
            'hypothyroid': {
                'fatigue': 0.2,
                'cold_sensitivity': 0.15,
                'weight_gain': 0.15,
                'dry_skin': 0.1,
                'constipation': 0.1,
                'depression': 0.1,
                'slow_heart_rate': 0.1,
                'muscle_weakness': 0.1
            },
            'hyperthyroid': {
                'anxiety': 0.2,
                'heat_sensitivity': 0.15,
                'weight_loss': 0.15,
                'tremors': 0.1,
                'rapid_heart_rate': 0.1,
                'sweating': 0.1,
                'insomnia': 0.1,
                'muscle_weakness': 0.1
            }
        }
    
    
    
    
    def calculate_symptom_score(self, responses):
        """
        Calculate symptom scores for each condition.
        
        Args:
            responses (dict): Dictionary of symptom responses
            
        Returns:
            dict: Scores for each condition
        """
        scores = {
            'hypothyroid': 0.0,
            'hyperthyroid': 0.0
        }
        
        for condition, symptoms in self.symptom_weights.items():
            for symptom, weight in symptoms.items():
                if symptom in responses and responses[symptom]:
                    scores[condition] += weight
        
        return scores
    
    def predict(self, hormone_values, symptom_responses):
        """
        Make prediction using hormone values and symptoms.
        
        Args:
            hormone_values (dict): Dictionary of hormone test values
            symptom_responses (dict): Dictionary of symptom responses
            
        Returns:
            dict: Prediction results
        """
        logging.info("Making hormone-based prediction")
        try:
            logging.info("Running hormone-based prediction")

            # Convert input to scaled features
            df_input = pd.DataFrame([hormone_values])

            # Get prediction and probabilities
            pred_probs = self.pipeline.predict_proba(df_input)[0]
            pred_class = np.argmax(pred_probs)

            # Symptom score
            symptom_scores = self._calculate_symptom_score(symptom_responses)

            # Combine score
            combined_score = pred_probs[pred_class] * 0.7 + max(symptom_scores.values()) * 0.3

            condition_map = {0: 'normal', 1: 'hypothyroid', 2: 'hyperthyroid'}
            predicted_condition = condition_map[pred_class]

            # Risk level logic
            risk_level = (
                'high' if combined_score >= 0.7 else
                'medium' if combined_score >= 0.4 else
                'low'
            )

            result = {
                'condition': predicted_condition,
                'confidence': float(combined_score),
                'risk_level': risk_level,
                'recommendation': self._generate_recommendation(predicted_condition, risk_level),
                'hormone_prediction': {
                    'class': predicted_condition,
                    'confidence': float(pred_probs[pred_class]),
                    'probabilities': {
                        condition: float(prob)
                        for condition, prob in zip(condition_map.values(), pred_probs)
                    }
                },
                'symptom_scores': symptom_scores
            }

            logging.info(f"Prediction completed: {result}")
            return result

        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise
    
    def _generate_recommendation(self, condition, risk_level):
        """
        Generate recommendation based on condition and risk level.
        
        Args:
            condition (str): Predicted condition
            risk_level (str): Risk level
            
        Returns:
            str: Recommendation text
        """
        if risk_level == 'low':
            return "No immediate action needed. Monitor for any changes in symptoms."
        elif risk_level == 'medium':
            return f"Consider consulting a doctor for {condition} evaluation."
        else:
            return f"Strongly recommend immediate consultation with a doctor for {condition} evaluation and possible thyroid function tests."
    
    def save_model(self, model_path):
        """
        Save the trained model and scaler.
        
        Args:
            model_path (str): Path to save the model
        """
        logging.info(f"Saving model to {model_path}")
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, model_path)
            logging.info("Model saved successfully")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path):
        """
        Load a trained pipeline (preprocessing + model).
        
        Args:
            model_path (str): Path to load the model from
        """
        logging.info(f"Loading model pipeline from {model_path}")
        try:
            self.model = joblib.load(model_path)
            self.scaler = None  # Not needed, handled in pipeline
            logging.info("Model pipeline loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

            
    def initialize_default_model(self):
        """
        Initialize a default model and scaler with sample data.
        This is used when no pre-trained model is available.
        """
        logging.info("Initializing default hormone model")
        try:
            # Create sample data for fitting the scaler
            sample_data = np.array([
                [2.5, 150, 8.0],  # Normal values
                [10.0, 60, 3.0],   # Hypothyroid values
                [0.1, 250, 15.0]   # Hyperthyroid values
            ])
            
            # Fit the scaler with sample data
            self.scaler.fit(sample_data)
            
            # Initialize a basic model
            self.model = RandomForestClassifier(
                n_estimators=10,
                max_depth=3,
                random_state=42
            )
            
            # Create sample labels (0: normal, 1: hypothyroid, 2: hyperthyroid)
            sample_labels = np.array([0, 1, 2])
            
            # Fit the model with scaled sample data
            sample_data_scaled = self.scaler.transform(sample_data)
            self.model.fit(sample_data_scaled, sample_labels)
            
            logging.info("Default hormone model initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing default model: {str(e)}")
            raise 