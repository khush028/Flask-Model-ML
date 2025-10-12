from flask import Flask, request, jsonify
import joblib
import pandas as pd

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
        data = request.get_json()

        # Convert JSON to DataFrame for prediction
        df = pd.DataFrame([data])

        # Debug: print input data
        print("Input data for prediction:")
        print(df.head())

        # Make prediction (0 or 1)
        prediction = model.predict(df)[0]

        # Debug: print prediction result
        print("Prediction result:", prediction)

        # Return response as JSON
        return jsonify({'failure_prediction': int(prediction)})
    except Exception as e:
        # Return error message for debugging
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run app on all IP addresses on port 5000 with debug on
    app.run(host='0.0.0.0', port=5000, debug=True)
