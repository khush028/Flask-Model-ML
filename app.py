from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained ML model
model = joblib.load("vehicle_failure_model.pkl")

# Health check
@app.route("/", methods=["GET"])
def home():
    return "✅ Vehicle Failure Prediction API is running"

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1️⃣ Read JSON input
        data = request.get_json(force=True)

        # 2️⃣ Create DataFrame in SAME ORDER as training
        df = pd.DataFrame([[
            float(data["Engine_Temperature"]),
            float(data.get("Mileage", 0)),
            float(data["Oil_Pressure"]),
            float(data["Battery_Voltage"]),
            float(data["Vehicle_Speed"])
        ]], columns=[
            "Engine_Temperature",
            "Mileage",
            "Oil_Pressure",
            "Battery_Voltage",
            "Vehicle_Speed"
        ])

        # 3️⃣ Predict probability
        proba = model.predict_proba(df)[0][1]
        confidence = round(float(proba), 4)

        # 4️⃣ Failure decision (demo threshold)
        failure_prediction = 1 if confidence >= 0.6 else 0

        # 5️⃣ Failure Type (Rule-based RCA)
        failure_type = "No_Failure"

        if failure_prediction == 1:
            if df.loc[0, "Engine_Temperature"] > 110:
                failure_type = "Engine_Overheating"
            elif df.loc[0, "Oil_Pressure"] < 6:
                failure_type = "Oil_System_Failure"
            elif df.loc[0, "Battery_Voltage"] < 11.5:
                failure_type = "Battery_Failure"
            elif df.loc[0, "Vehicle_Speed"] > 100:
                failure_type = "Transmission_Stress"
            else:
                failure_type = "General_Mechanical_Risk"

        # 6️⃣ Risk Level mapping
        if confidence >= 0.85:
            risk_level = "High"
        elif confidence >= 0.6:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # 7️⃣ Final response
        return jsonify({
            "Failure_Prediction": failure_prediction,
            "Failure_Type": failure_type,
            "Risk_Level": risk_level,
            "Confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
