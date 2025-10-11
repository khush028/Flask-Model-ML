from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model (make sure the 'vehicle_failure_model.pkl' file is in the same folder)
model = joblib.load('vehicle_failure_model.pkl')

# Test route to verify server is running
@app.route('/')
def home():
    return "API is running"

# Prediction route, accepts POST requests only
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        port = int(os.environ.get("PORT", 5000))
        data = request.get_json()

        # Convert JSON to DataFrame for prediction
        df = pd.DataFrame([data])

        # Make prediction (0 or 1)
        prediction = model.predict(df)[0]

        # Return response as JSON
        return jsonify({'failure_prediction': int(prediction)})
    except Exception as e:
        # Return error message for debugging
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run app on all IP addresses on port 5000 with debug on
    port = int(os.environ.get("PORT" , 5000))
    app.run(host='0.0.0.0', port=5000, debug=True)

