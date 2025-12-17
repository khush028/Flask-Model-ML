import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# LOAD DATASET
# ---------------------------
df = pd.read_csv("vehicle_data.csv")  
# columns:
# Engine_Temperature, Mileage, Oil_Pressure,
# Battery_Voltage, Vehicle_Speed, Failure, Failure_Type

# ---------------------------
# FEATURES
# ---------------------------
X = df[
    ["Engine_Temperature", "Mileage", "Oil_Pressure",
     "Battery_Voltage", "Vehicle_Speed"]
].values

# ---------------------------
# MODEL 1: FAILURE YES / NO
# ---------------------------
y_failure = df["Failure"].values

failure_model = RandomForestClassifier(n_estimators=100, random_state=42)
failure_model.fit(X, y_failure)

joblib.dump(failure_model, "vehicle_failure_model.pkl")

# ---------------------------
# MODEL 2: FAILURE TYPE
# ---------------------------
le = LabelEncoder()
df["Failure_Type_Encoded"] = le.fit_transform(df["Failure_Type"])

y_type = df["Failure_Type_Encoded"].values

type_model = RandomForestClassifier(n_estimators=100, random_state=42)
type_model.fit(X, y_type)

joblib.dump(type_model, "failure_type_model.pkl")
joblib.dump(le, "failure_type_encoder.pkl")

print("✅ Models trained & saved successfully")
