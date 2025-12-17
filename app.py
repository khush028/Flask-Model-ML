from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("vehicle_failure_model.pkl")

FEATURE_ORDER = [
    "Engine_Temperature",
    "Mileage",
    "Oil_Pressure",
    "Battery_Voltage",
    "Vehicle_Speed"
]

@app.route("/")
def home():
    return "Vehicle Failure Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Build row in EXACT training order
        row = [
            float(data["Engine_Temperature"]),
            float(data.get("Mileage", 0)),
            float(data["Oil_Pressure"]),
            float(data["Battery_Voltage"]),
            float(data["Vehicle_Speed"])
        ]

        df = pd.DataFrame([row], columns=FEATURE_ORDER)

        proba = model.predict_proba(df)[0][1]
        confidence = round(float(proba), 2)

        failure_prediction = 1 if confidence >= 0.6 else 0

        if failure_prediction == 1:
            if df.loc[0, "Engine_Temperature"] > 110:
                failure_type = "Engine_Overheating"
            elif df.loc[0, "Oil_Pressure"] < 6:
                failure_type = "Oil_System_Failure"
            elif df.loc[0, "Battery_Voltage"] < 11.5:
                failure_type = "Battery_Failure"
            else:
                failure_type = "General_Mechanical_Risk"
        else:
            failure_type = "No_Failure"

        if confidence >= 0.85:
            risk_level = "High"
        elif confidence >= 0.6:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return jsonify({
            "Failure_Prediction": failure_prediction,
            "Failure_Type": failure_type,
            "Risk_Level": risk_level,
            "Confidence": confidence
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
