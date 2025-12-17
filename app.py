from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("vehicle_failure_model.pkl")

FEATURE_ORDER = [
    "Engine_Temperature",
    "Oil_Pressure",
    "Battery_Voltage",
    "Vehicle_Speed"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        row = [
            float(data["Engine_Temperature"]),
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

        risk_level = "High" if confidence >= 0.85 else "Medium" if confidence >= 0.6 else "Low"

        return jsonify({
            "Failure_Prediction": failure_prediction,
            "Failure_Type": failure_type,
            "Risk_Level": risk_level,
            "Confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
