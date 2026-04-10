import os
from predict import ThyroidScreeningSystem
import logging

def get_user_input():
    """
    Get user input for screening.
    """
    print("\n=== Thyroid Screening System ===")
    
    # Get image path
    while True:
        image_path = input("\nEnter path to neck image: ")
        if os.path.exists(image_path):
            break
        print("Invalid path. Please try again.")
    
    # Get symptom responses
    print("\nAnswer the following symptoms (yes/no):")
    symptoms = {
        'fatigue': input("Do you experience unusual fatigue? ").lower() == 'yes',
        'cold_sensitivity': input("Do you have increased sensitivity to cold? ").lower() == 'yes',
        'weight_gain': input("Have you experienced unexplained weight gain? ").lower() == 'yes',
        'dry_skin': input("Do you have dry or rough skin? ").lower() == 'yes',
        'constipation': input("Do you experience constipation? ").lower() == 'yes',
        'depression': input("Have you been feeling unusually depressed? ").lower() == 'yes',
        'slow_heart_rate': input("Has anyone told you that you have a slow heart rate? ").lower() == 'yes',
        'muscle_weakness': input("Do you experience muscle weakness? ").lower() == 'yes',
        'anxiety': input("Do you experience unusual anxiety? ").lower() == 'yes',
        'heat_sensitivity': input("Do you have increased sensitivity to heat? ").lower() == 'yes',
        'weight_loss': input("Have you experienced unexplained weight loss? ").lower() == 'yes',
        'tremors': input("Do you experience tremors? ").lower() == 'yes',
        'rapid_heart_rate': input("Has anyone told you that you have a rapid heart rate? ").lower() == 'yes',
        'sweating': input("Do you experience excessive sweating? ").lower() == 'yes',
        'insomnia': input("Do you have trouble sleeping? ").lower() == 'yes'
    }
    
    return image_path, symptoms

def get_hormone_input():
    """
    Get all user input including symptoms, medical conditions, and hormone test values.
    Returns a dictionary suitable for model prediction.
    """
    print("\n=== Complete Patient Data Collection ===")
    print("🔍 Please enter the following details. Type 't' for true and 'f' for false where applicable.\n")

    try:
        input_data = {
            'age': int(input("Age: ")),
            'sex': input("Sex (M/F): "),
            'on_thyroxine': input("On Thyroxine? (t/f): "),
            'query_on_thyroxine': input("Query on Thyroxine? (t/f): "),
            'on_antithyroid_medication': input("On Antithyroid Medication? (t/f): "),
            'sick': input("Sick? (t/f): "),
            'pregnant': input("Pregnant? (t/f): "),
            'thyroid_surgery': input("Thyroid Surgery? (t/f): "),
            'I131_treatment': input("I131 Treatment? (t/f): "),
            'query_hypothyroid': input("Query Hypothyroid? (t/f): "),
            'query_hyperthyroid': input("Query Hyperthyroid? (t/f): "),
            'lithium': input("On Lithium? (t/f): "),
            'goitre': input("Goitre? (t/f): "),
            'tumor': input("Tumor? (t/f): "),
            'hypopituitary': input("Hypopituitary? (t/f): "),
            'psych': input("Psych condition? (t/f): "),
            'TSH_measured': 't',
            'T3_measured': 't',
            'TT4_measured': 't',
            'T4U_measured': 't',
            'FTI_measured': 't',
        }

        print("\n=== 📊 Hormone Test Input ===")
        print("🔬 Reference Ranges:\n  • TSH: 0.4–4.0 mIU/L\n  • T3: 80–200 ng/dL\n  • T4: 4.5–12.0 μg/dL\n")

        input_data['TSH'] = float(input("Enter TSH value (mIU/L): "))
        input_data['T3'] = float(input("Enter T3 value (ng/dL): "))
        input_data['TT4'] = float(input("Enter TT4 value (μg/dL): "))
        input_data['T4U'] = float(input("Enter T4U value: "))
        input_data['FTI'] = float(input("Enter FTI value: "))

    except ValueError:
        print("❌ Invalid input. Please enter numeric values correctly.")
        return get_hormone_input()

    return input_data


def display_stage1_results(results):
    """
    Display stage 1 screening results and determine if stage 2 should be offered.
    """
    print("\n=== Stage 1 Results ===")
    print(f"Predicted Condition: {results['final_condition']}")
    print(f"Combined Risk Score: {results['combined_risk_score']:.1f}%")
    print(f"Risk Level: {results['risk_level']}")
    print(f"Recommendation: {results['recommendation']}")
    
    # Check if combined score is between 25% and 39%
    combined_score = results['combined_risk_score']
    print(f"\nDebug: Combined score is {combined_score:.2f}%")  # Debug line to show the score
    
    if 25 <= combined_score < 39:
        print("\nFor more precise diagnosis, you may proceed to detailed hormone analysis.")
        print("This will help us provide a more accurate assessment of your condition.")
        proceed = input("Would you like to proceed to Stage 2? (yes/no): ").lower() == 'yes'
        return proceed
    else:
        if combined_score < 25:
            print("\nBased on the initial assessment, your risk level is low.")
            print("Hormone analysis is not required at this time.")
        else:
            print("\nBased on the initial assessment, your risk level is high.")
            print("Please consult with a healthcare provider for further evaluation.")
        return False

def display_final_results(results):
    """
    Display complete screening results.
    """
    print("\n=== Final Screening Results ===")
    print(f"Final Condition: {results['final_condition']}")
    print(f"Risk Score: {results['combined_risk_score']:.1f}%")
    print(f"Risk Level: {results['risk_level']}")
    print(f"Recommendation: {results['recommendation']}")
    
    print("\n=== Image Prediction Details ===")
    print(f"Predicted Class: {results['image_prediction']['class']}")
    print(f"Confidence: {results['image_prediction']['confidence']:.2f}")
    
    if results['hormone_results']:
        print("\n=== Hormone Test Results ===")
        print(f"Predicted Condition: {results['hormone_results']['condition']}")
        print(f"Confidence: {results['hormone_results']['confidence']:.2f}")
        print("\nSymptom Scores:")
        for condition, score in results['hormone_results']['symptom_scores'].items():
            print(f"{condition}: {score:.2f}")

def main():
    # Initialize screening system
    hormone_model_path = "models/best_thyroid_model_pipeline.pkl"
    screening_system = ThyroidScreeningSystem(hormone_model_path=hormone_model_path)
    
    # Get initial user input
    image_path, symptoms = get_user_input()
    
    # Perform stage 1 screening
    stage1_results = screening_system.screen_patient(
        image_path=image_path,
        questionnaire_responses=symptoms
    )
    
    # Display stage 1 results and check if stage 2 should be offered
    proceed_to_stage2 = display_stage1_results(stage1_results)
    
    if proceed_to_stage2:
        # Get hormone values
        hormone_values = get_hormone_input()
        
        # Perform complete screening with hormone data
        final_results = screening_system.screen_patient(
            image_path=image_path,
            questionnaire_responses=symptoms,
            hormone_values=hormone_values
        )
        
        # Display final results
        display_final_results(final_results)
    else:
        print("\nScreening completed with initial results.")

if __name__ == "__main__":
    main() 