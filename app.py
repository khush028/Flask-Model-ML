from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("vehicle_failure_model.pkl")

@app.route("/")
def home():
    return "Vehicle Failure Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Accept dict or list
        if isinstance(data, list):
            data = data[0]

        # 🔥 EXACT ORDER USED DURING TRAINING
        feature_array = np.array([[
            float(data.get("Engine_Temperature", 0)),
            float(data.get("Mileage", 0)),
            float(data.get("Oil_Pressure", 0)),
            float(data.get("Battery_Voltage", 0)),
            float(data.get("Vehicle_Speed", 0))
        ]])

        prediction = model.predict(feature_array)[0]
        confidence = model.predict_proba(feature_array).max()

        return jsonify({
            "Failure_Prediction": int(prediction),
            "Confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
