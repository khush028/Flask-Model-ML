from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load ML model (binary failure prediction)
failure_model = joblib.load("vehicle_failure_model.pkl")


@app.route("/", methods=["GET"])
def home():
    return "Vehicle Failure Prediction API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Handle n8n body-wrapper
        if isinstance(data, dict) and "body" in data:
            data = data["body"]

        # -----------------------------
        # Prepare input (ORDER MATTERS)
        # -----------------------------
        features = np.array([[
            float(data.get("Engine_Temperature", 0)),
            float(data.get("Mileage", 0)),
            float(data.get("Oil_Pressure", 0)),
            float(data.get("Battery_Voltage", 0)),
            float(data.get("Vehicle_Speed", 0))
        ]])

        # -----------------------------
        # FAILURE PREDICTION (ML)
        # -----------------------------
        failure_pred = int(failure_model.predict(features)[0])
        confidence = float(failure_model.predict_proba(features).max())

        # -----------------------------
        # RISK LEVEL (LOGIC-BASED)
        # -----------------------------
        if confidence >= 0.85:
            risk_level = "High"
        elif confidence >= 0.60:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # -----------------------------
        # FAILURE TYPE (RULE-BASED)
        # -----------------------------
        if failure_pred == 1:
            if data.get("Engine_Temperature", 0) > 100:
                failure_type = "Engine"
            elif data.get("Battery_Voltage", 0) < 11.5:
                failure_type = "Battery"
            elif data.get("Oil_Pressure", 0) < 6:
                failure_type = "Oil"
            else:
                failure_type = "General"
        else:
            failure_type = "No_Failure"

        # -----------------------------
        # FINAL RESPONSE
        # -----------------------------
        response = {
            "Failure_Prediction": failure_pred,
            "Failure_Type": failure_type,
            "Risk_Level": risk_level,
            "Confidence": confidence
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
