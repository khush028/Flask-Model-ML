from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained ML model
model = joblib.load("vehicle_failure_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Vehicle Failure Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON from request
        data = request.get_json()

        # 🔥 HANDLE n8n BODY WRAPPER
        # If payload is { "body": { ... } }, unwrap it
        if isinstance(data, dict) and "body" in data:
            data = data["body"]

        # 🔥 CREATE NUMPY ARRAY IN EXACT TRAINING ORDER
        features = np.array([[
            float(data.get("Engine_Temperature", 0)),
            float(data.get("Mileage", 0)),
            float(data.get("Oil_Pressure", 0)),
            float(data.get("Battery_Voltage", 0)),
            float(data.get("Vehicle_Speed", 0))
        ]])

        # Prediction
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features).max()

        return jsonify({
            "Failure_Prediction": int(prediction),
            "Confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
