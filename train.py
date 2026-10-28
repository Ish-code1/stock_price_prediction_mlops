import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path

data = pd.read_csv('data.csv')

print(data.info())
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Handle missing values by dropping rows with NaN in critical columns
data_cleaned = data.dropna(subset=['Close', 'Prev Close'])

# Fill remaining missing numerical values with median
for col in ['Trades', 'Deliverable Volume', '%Deliverble']:
    data_cleaned[col] = data_cleaned[col].fillna(data_cleaned[col].median())

# Define Features (X) and Target (y)
# Predicting tomorrow's 'Close' price using today's numerical metrics
features = ['Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble']
X = data_cleaned[features]
y = data_cleaned['Close'].shift(-1).fillna(data_cleaned['Close'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Features and Target successfully prepared and split!")
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Initialize the Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 2. Train the model
print("Training the Random Forest Regressor...")
model.fit(X_train, y_train)

# 3. Make predictions
y_pred = model.predict(X_test)

# 4. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTraining Complete!")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")

import matplotlib.pyplot as plt

# Plot Actual vs Predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label='Actual Price', color='blue', alpha=0.7)
plt.plot(y_pred[:100], label='Predicted Price', color='orange', linestyle='--', alpha=0.9)
plt.title('Actual vs Predicted Stock Close Prices (First 100 Samples)')
plt.xlabel('Samples')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np

# Get feature importances from our trained Random Forest model
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature Importances for Predicting Tomorrow's Close Price")
plt.bar(range(X.shape[1]), importances[indices], align="center", color="teal")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha="right")
plt.xlim([-1, X.shape[1]])
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()


# Save the trained model to a file
model_dir = Path(__file__).resolve().parent / 'models'
model_dir.mkdir(exist_ok=True)
model_filename = model_dir / 'stock_price_predictor.joblib'
joblib.dump(model, model_filename)

print(f"Model successfully saved to '{model_filename}'!")

# Quick verification of the file creation
if Path(model_filename).exists():
    print(f"Verified: '{model_filename}' exists in your current workspace directory ({os.path.getsize(model_filename) / (1024*1024):.2f} MB).")