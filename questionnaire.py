import numpy as np
import config
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('thyroid_screening.log'),
        logging.StreamHandler()
    ]
)
class ThyroidQuestionnaire:
    def __init__(self):
        self.weights = config.QUESTIONNAIRE_WEIGHTS
        self.thresholds = config.RISK_THRESHOLDS
    
    def calculate_symptom_score(self, responses):
        """
        Calculate symptom scores for each condition based on questionnaire responses.
        
        Args:
            responses (dict): Dictionary of symptom responses (True/False)
            
        Returns:
            dict: Scores for each condition
        """
        scores = {
            'hypothyroidism': 0.0,
            'nodules': 0.0
        }
        
        # Calculate scores for each condition
        for condition, symptoms in self.weights.items():
            for symptom, weight in symptoms.items():
                if symptom in responses and responses[symptom]:
                    scores[condition] += weight
        
        return scores
    
    def get_risk_level(self, score):
        """
        Determine risk level based on score.
        
        Args:
            score (float): Combined risk score (0-1)
            
        Returns:
            str: Risk level ('low', 'medium', 'high')
        """
        logging.info(score)
        if score >= 0.4:
            return 'high'
        elif score >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def get_recommendation(self, risk_level, condition):
        """
        Generate recommendation based on risk level and condition.
        
        Args:
            risk_level (str): Risk level ('low', 'medium', 'high')
            condition (str): Predicted condition
            
        Returns:
            str: Recommendation text
        """
        if risk_level == 'low':
            return "No immediate action needed. Monitor for any changes in symptoms."
        elif risk_level == 'medium':
            return f"Consider consulting a doctor for {condition} evaluation."
        else:
            return f"Strongly recommend immediate consultation with a doctor for {condition} evaluation and possible thyroid function tests."
    
    def process_responses(self, responses):
        """
        Process questionnaire responses and generate comprehensive results.
        
        Args:
            responses (dict): Dictionary of symptom responses
            
        Returns:
            dict: Comprehensive results including scores, risk level, and recommendation
        """
        # Calculate symptom scores
        scores = self.calculate_symptom_score(responses)
        
        # Get primary condition (highest score)
        primary_condition = max(scores, key=scores.get)
        primary_score = scores[primary_condition]
        
        # Calculate overall risk score (0-100%)
        risk_score = primary_score * 100
        
        # Get risk level
        risk_level = self.get_risk_level(primary_score)
        
        # Get recommendation
        recommendation = self.get_recommendation(risk_level, primary_condition)
        
        return {
            'risk_score': risk_score,
            'primary_condition': primary_condition,
            'symptom_scores': scores,
            'risk_level': risk_level,
            'recommendation': recommendation
        }

# Example usage:
if __name__ == "__main__":
    # Example responses
    example_responses = {
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
    
    # Process responses
    questionnaire = ThyroidQuestionnaire()
    results = questionnaire.process_responses(example_responses)
    
    # Print results
    print("\nQuestionnaire Results:")
    print(f"Risk Score: {results['risk_score']:.1f}%")
    print(f"Primary Condition: {results['primary_condition']}")
    print(f"Risk Level: {results['risk_level']}")
    print(f"Recommendation: {results['recommendation']}")
    
    print("\nDetailed Symptom Scores:")
    for condition, score in results['symptom_scores'].items():
        print(f"{condition}: {score:.2f}") 