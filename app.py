from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('vehicle_failure_model.pkl')

# Test route
@app.route('/')
def home():
    return "Vehicle Failure Prediction API is running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Handle both dict and list input
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])

        # 🔥 FORCE SAME FEATURE ORDER AS TRAINING
        df = df[
            [
                'Engine_Temperature',
                'Mileage',
                'Oil_Pressure',
                'Battery_Voltage',
                'Vehicle_Speed'
            ]
        ]

        print("Input data used for prediction:")
        print(df)

        # Make prediction
        prediction = model.predict(df)[0]
        confidence = model.predict_proba(df).max()

        # Return response
        return jsonify({
            "Failure_Prediction": int(prediction),
            "Confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
