from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Load Trained Components ---
try:
    pipeline = joblib.load('stroke_prediction_pipeline.joblib')
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    pipeline = None

try:
    training_columns = joblib.load('training_columns.joblib')
    print("Training columns loaded successfully.")
    print("Expected columns:", training_columns)
except Exception as e:
    print(f"Error loading training columns: {e}")
    training_columns = None

try:
    median_bmi = joblib.load('median_bmi.joblib')
    print(f"Median BMI for imputation loaded successfully: {median_bmi}")
except Exception as e:
    print(f"Error loading median BMI: {e}")
    median_bmi = 28.1 # A common default if loading fails, adjust if needed

# --- Routes ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives user input, preprocesses, predicts, and returns result with risk level."""
    if not pipeline or not training_columns:
        return jsonify({'error': 'Model components not loaded properly. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)
        print("Received data:", data)

        input_df = pd.DataFrame([data])

        # --- Data Preprocessing ---
        if 'bmi' not in input_df.columns or input_df['bmi'].iloc[0] is None or str(input_df['bmi'].iloc[0]).strip() == '':
             print(f"BMI is missing or empty. Imputing with median: {median_bmi}")
             input_df['bmi'] = median_bmi
        else:
            try:
                input_df['bmi'] = pd.to_numeric(input_df['bmi'])
            except ValueError:
                 print(f"Invalid BMI value provided. Imputing with median: {median_bmi}")
                 input_df['bmi'] = median_bmi

        try:
            input_df['age'] = pd.to_numeric(input_df['age'])
            input_df['avg_glucose_level'] = pd.to_numeric(input_df['avg_glucose_level'])
            input_df['hypertension'] = pd.to_numeric(input_df['hypertension'])
            input_df['heart_disease'] = pd.to_numeric(input_df['heart_disease'])
        except ValueError as e:
             print(f"Error converting numeric fields: {e}")
             return jsonify({'error': f'Invalid numeric value provided. {e}'}), 400

        try:
             input_df = input_df[training_columns]
             print("DataFrame columns aligned with training columns.")
        except KeyError as e:
            print(f"Missing expected column in input: {e}")
            return jsonify({'error': f'Missing input field: {e}'}), 400

        # --- Prediction ---
        print("DataFrame before prediction:\n", input_df.to_string())

        prediction_proba = pipeline.predict_proba(input_df)[0]
        stroke_probability = prediction_proba[1] # Probability of class 1 (stroke)

        print(f"Prediction Probability (Stroke): {stroke_probability:.4f}")

        # --- Determine Risk Level based on Probability ---
        # ** Adjust these thresholds as needed based on model performance and requirements **
        if stroke_probability < 0.15: # Example Threshold for Low Risk
            risk_level = 'low'
            prediction_text = 'Low Risk of Stroke'
        elif stroke_probability < 0.50: # Example Threshold for Medium Risk
            risk_level = 'medium'
            prediction_text = 'Medium Risk of Stroke'
        else: # High Risk
            risk_level = 'high'
            prediction_text = 'High Risk of Stroke'

        print(f"Determined Risk Level: {risk_level.upper()}")

        # --- Format Output ---
        # Get the binary prediction as well if needed (based on default 0.5 threshold)
        prediction_code = int(pipeline.predict(input_df)[0])

        result = {
            'prediction_code': prediction_code,
            'prediction_text': prediction_text,
            'risk_level': risk_level, # Send 'low', 'medium', or 'high'
            'probability_stroke': round(stroke_probability * 100, 2) # Percentage
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)