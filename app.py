from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models
failure_model = joblib.load("vehicle_failure_model.pkl")

# (Optional) Failure type model
# Agar tumne failure_type_model train kiya hai tabhi use hoga
try:
    type_model = joblib.load("failure_type_model.pkl")
    type_encoder = joblib.load("failure_type_encoder.pkl")
    FAILURE_TYPE_ENABLED = True
except:
    FAILURE_TYPE_ENABLED = False


@app.route("/", methods=["GET"])
def home():
    return "Vehicle Failure Prediction API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # n8n body-wrapper handle
        if isinstance(data, dict) and "body" in data:
            data = data["body"]

        # ---- INPUT ORDER MUST MATCH TRAINING ----
        features = np.array([[
            float(data.get("Engine_Temperature", 0)),
            float(data.get("Mileage", 0)),
            float(data.get("Oil_Pressure", 0)),
            float(data.get("Battery_Voltage", 0)),
            float(data.get("Vehicle_Speed", 0))
        ]])

        # ---- FAILURE PREDICTION ----
        failure_pred = int(failure_model.predict(features)[0])
        confidence = float(failure_model.predict_proba(features).max())

        # ---- RISK LEVEL LOGIC (NEW) ----
        if confidence >= 0.85:
            risk_level = "High"
        elif confidence >= 0.60:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        response = {
            "Failure_Prediction": failure_pred,
            "Confidence": confidence,
            "Risk_Level": risk_level
        }

        # ---- FAILURE TYPE (OPTIONAL BUT ADDED) ----
        if FAILURE_TYPE_ENABLED and failure_pred == 1:
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
