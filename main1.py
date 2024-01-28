import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the synthetic dataset
csv_path = r'D:\ev-tech\synthetic_ev_dataset.csv'
data = pd.read_csv(csv_path)

# Define features (X) and target variable (y)
features = ['Temperature', 'Initial_SoH', 'Final_SoH', 'Initial_SoC', 'Final_SoC', 'Class']
target = 'Charging_Cycles'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model for future use
model_path = r'D:\ev-tech\charging_cycles_prediction_model.joblib'
joblib.dump(model, model_path)
