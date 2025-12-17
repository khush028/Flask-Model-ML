from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models
failure_model = joblib.load("vehicle_failure_model.pkl")
type_model = joblib.load("failure_type_model.pkl")
type_encoder = joblib.load("failure_type_encoder.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Vehicle Failure Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Handle n8n body wrapper
        if isinstance(data, dict) and "body" in data:
            data = data["body"]

        # Create NumPy array (EXACT ORDER)
        features = np.array([[
            float(data.get("Engine_Temperature", 0)),
            float(data.get("Mileage", 0)),
            float(data.get("Oil_Pressure", 0)),
            float(data.get("Battery_Voltage", 0)),
            float(data.get("Vehicle_Speed", 0))
        ]])

        # Failure prediction
        failure_pred = failure_model.predict(features)[0]
        failure_confidence = failure_model.predict_proba(features).max()

        response = {
            "Failure_Prediction": int(failure_pred),
            "Confidence": float(failure_confidence)
        }

        # If failure predicted, also predict failure type
        if failure_pred == 1:
            type_pred = type_model.predict(features)[0]
            failure_type = type_encoder.inverse_transform([type_pred])[0]
            response["Failure_Type"] = failure_type
        else:
            response["Failure_Type"] = "No_Failure"

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
