# Filename: app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for requests from file:/// protocol

# --- Load Model and Preprocessing Info ---
# Assumes .pkl files are in the same directory
try:
    pipe = joblib.load('car_price_model.pkl')
    # Load companies and fuel_types for validation if needed
    companies = joblib.load('companies.pkl')
    fuel_types = joblib.load('fuel_types.pkl')
    print("Model pipeline and supporting lists loaded successfully.")
except FileNotFoundError:
    print("Error: Model/supporting files (.pkl) not found.")
    print("Please run train_model.py first to generate them.")
    pipe = None
    companies = []
    fuel_types = []
except Exception as e:
    print(f"An error occurred loading the model or lists: {e}")
    pipe = None
    companies = []
    fuel_types = []

# --- API Route for Prediction ---

# Removed the '/' route that used render_template, as index.html is opened directly

@app.route('/predict', methods=['POST'])
def predict():
    if pipe is None:
        return jsonify({'error': 'Model not loaded. Cannot predict.'}), 500

    try:
        data = request.get_json()
        if not data:
             return jsonify({'error': 'No JSON data received'}), 400
        print("Received data:", data)

        # Extract data
        company = data.get('company')
        year = data.get('year')
        kms_driven = data.get('kms_driven')
        fuel_type = data.get('fuel_type')

        # --- Basic Input Validation ---
        if not all([company, year, kms_driven, fuel_type]):
             return jsonify({'error': 'Missing required input fields (company, year, kms_driven, fuel_type)'}), 400

        try:
            year = int(year)
            kms_driven = int(kms_driven)
        except (ValueError, TypeError):
            return jsonify({'error': 'Year and Kms Driven must be valid numbers'}), 400

        if year <= 1980 or year > 2025: # Adjust max year if needed
             return jsonify({'error': 'Invalid year provided'}), 400
        if kms_driven < 0:
             return jsonify({'error': 'Kilometers driven cannot be negative'}), 400
        # Optional: More robust validation against loaded lists
        if company not in companies:
             print(f"Warning: Company '{company}' not in training data companies list.")
             # Decide if you want to return an error or let the model handle it (due to handle_unknown='ignore')
             # return jsonify({'error': f'Invalid company: {company}. Not found in training data.'}), 400
        if fuel_type not in fuel_types:
             print(f"Warning: Fuel type '{fuel_type}' not in training data fuel types list.")
             # return jsonify({'error': f'Invalid fuel type: {fuel_type}. Not found in training data.'}), 400


        # --- Prepare Input for Model ---
        input_data = pd.DataFrame([[company, year, kms_driven, fuel_type]],
                                  columns=['company', 'year', 'kms_driven', 'fuel_type'])
        print("Prepared input DataFrame:\n", input_data)

        # --- Make Prediction ---
        prediction = pipe.predict(input_data)
        predicted_price = max(0, prediction[0]) # Ensure price is not negative

        print(f"Prediction result: {predicted_price:.2f}")

        # Return prediction as JSON
        return jsonify({'predicted_price': round(predicted_price, 2)})

    except Exception as e:
        print(f"Error during prediction processing: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed debugging traceback
        return jsonify({'error': 'An internal error occurred during prediction.'}), 500


if __name__ == '__main__':
    # Host '0.0.0.0' makes it accessible on your local network
    # Host '127.0.0.1' (default) makes it accessible only from your own machine
    # Debug=True is helpful for development, turn off for 'production'
    app.run(debug=True, host='127.0.0.1', port=5000)