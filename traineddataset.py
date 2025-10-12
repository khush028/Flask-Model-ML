import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib  # To save the model


# Load dataset
data = pd.read_csv('vehicle_training_dataset.csv')

# Check column names (make sure target column is 'Failure_Status' as generated)
print(data.columns)

# Select features and target
features = ['Engine_Temperature','Mileage', 'Oil_Pressure', 'Battery_Voltage', 'Vehicle_Speed']
X = data[features]
y = data['Failure_Prediction']  # Use the correct target column name

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increased iterations for convergence
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f'Model accuracy: {accuracy:.2f}')

# Save the trained model for deployment
joblib.dump(model, 'vehicle_failure_model.pkl')  # Remove extra usage of assignment (=)
