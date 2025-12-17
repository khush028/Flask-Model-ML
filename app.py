from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained ML model
model = joblib.load("vehicle_failure_model.pkl")

# Health check route
@app.route("/", methods=["GET"])
def home():
    return "Vehicle Failure Prediction API is running"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Handle dict OR list input
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])

        # 🔥 REQUIRED FEATURES (EXACT TRAINING ORDER)
        required_features = [
            "Engine_Temperature",
            "Mileage",
            "Oil_Pressure",
            "Battery_Voltage",
            "Vehicle_Speed"
        ]

        # Add missing features with default value 0
        for col in required_features:
            if col not in df.columns:
                df[col] = 0

        # Convert all values to numeric (safety)
        df = df[required_features].apply(pd.to_numeric, errors="coerce").fillna(0)

        print("Final DataFrame for prediction:")
        print(df)

        # Prediction
        prediction = model.predict(df)[0]
        confidence = model.predict_proba(df).max()

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
