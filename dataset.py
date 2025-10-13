# --- Vehicle Predictive Maintenance Dataset Generator ---
import pandas as pd
import numpy as np

# Reproducible random numbers
np.random.seed(42)

# Number of vehicle records
n = 500

# Create synthetic sensor data
data = {
    "Engine_Temperature": np.random.normal(95, 10, n),
    "Mileage": np.random.randint(10000, 150000, n),
    "Oil_Pressure": np.random.normal(7.5, 1.0, n),        
    "Battery_Voltage": np.random.normal(12.0, 0.5, n),    
    "Vehicle_Speed": np.random.randint(0, 120, n)          
}

# Simple failure rule for demo
data["Failure_Prediction"] = (
    (data["Engine_Temperature"] > 110) | 
    (data["Oil_Pressure"] < 6) | 
    (data["Battery_Voltage"] < 11.5)
).astype(int)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV file
df.to_csv("synthetic_vehicle_data.csv", index=False)

print("✅ synthetic_vehicle_data.csv file generated successfully!")
print(df.head())


