from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("vehicle_failure_model.pkl")

# Root check
@app.route("/")
def home():
    return "Vehicle Failure Prediction API is running"

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1️⃣ Get input JSON
        data = request.get_json()

        # 2️⃣ Force DataFrame with SAME ORDER as training
        df = pd.DataFrame([{
            "Engine_Temperature": float(data["Engine_Temperature"]),
            "Mileage": float(data.get("Mileage", 0)),
            "Oil_Pressure": float(data["Oil_Pressure"]),
            "Battery_Voltage": float(data["Battery_Voltage"]),
            "Vehicle_Speed": float(data["Vehicle_Speed"])
        }])

        # 3️⃣ Prediction probability
        proba = model.predict_proba(df)[0][1]
        confidence = round(float(proba), 4)

        # 4️⃣ CONFIDENCE-BASED FAILURE (DEMO SAFE)
        failure_prediction = 1 if confidence >= 0.6 else 0

        # 5️⃣ FAILURE TYPE LOGIC (RULE-BASED RCA)
        failure_type = "No_Failure"

        if failure_prediction == 1:
            if df["Engine_Temperature"][0] > 110:
                failure_type = "Engine_Overheating"
            elif df["Oil_Pressure"][0] < 6:
                failure_type = "Oil_System_Failure"
            elif df["Battery_Voltage"][0] < 11.5:
                failure_type = "Battery_Failure"
            elif df["Vehicle_Speed"][0] > 100:
                failure_type = "Transmission_Stress"
            else:
                failure_type = "General_Mechanical_Risk"

        # 6️⃣ RISK LEVEL MAPPING
        if confidence >= 0.85:
            risk_level = "High"
        elif confidence >= 0.6:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # 7️⃣ FINAL RESPONSE
        response = {
            "Failure_Prediction": failure_prediction,
            "Failure_Type": failure_type,
            "Risk_Level": risk_level,
            "Confidence": confidence
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
