from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("vehicle_failure_model.pkl")

@app.route("/")
def home():
    return "Vehicle Failure Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        df = pd.DataFrame([{
            "Engine_Temperature": float(data["Engine_Temperature"]),
            "Mileage": float(data.get("Mileage", 0)),
            "Oil_Pressure": float(data["Oil_Pressure"]),
            "Battery_Voltage": float(data["Battery_Voltage"]),
            "Vehicle_Speed": float(data["Vehicle_Speed"])
        }])

        proba = model.predict_proba(df)[0][1]
        confidence = round(float(proba), 2)

        failure_prediction = 1 if confidence >= 0.6 else 0

        failure_type = "No_Failure"
        if failure_prediction == 1:
            if df["Engine_Temperature"][0] > 110:
                failure_type = "Engine_Overheating"
            elif df["Oil_Pressure"][0] < 6:
                failure_type = "Oil_System_Failure"
            elif df["Battery_Voltage"][0] < 11.5:
                failure_type = "Battery_Failure"
            else:
                failure_type = "General_Mechanical_Risk"

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
