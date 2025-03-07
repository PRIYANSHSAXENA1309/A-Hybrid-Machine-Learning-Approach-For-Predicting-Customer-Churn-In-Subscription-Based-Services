import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

new_data = pd.DataFrame({
    'CustomerID': [9999], 
    'SubscriptionType': ['Premium'],
    'PaymentMethod': ['Credit Card'],
    'PaperlessBilling': ['Yes'],
    'ContentType': ['Movies'],
    'MultiDeviceAccess': ['Yes'],
    'DeviceRegistered': ['Smartphone'],
    'GenrePreference': ['Action'],
    'Gender': ['Male'],
    'ParentalControl': ['No'],
    'SubtitlesEnabled': ['Yes'],
    'AccountAge': [24], 
    'MonthlyCharges': [19.99],
    'TotalCharges': [479.76],
    'ViewingHoursPerWeek': [15],
    'SupportTicketsPerMonth': [0],
    'AverageViewingDuration': [1.5],
    'ContentDownloadsPerMonth': [2],
    'UserRating': [4],
    'WatchlistSize': [20]
})


label_encoder = LabelEncoder()
for col in ['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'ContentType', 'MultiDeviceAccess',
            'DeviceRegistered', 'GenrePreference', 'Gender', 'ParentalControl', 'SubtitlesEnabled']:
    new_data[col] = label_encoder.fit_transform(new_data[col])

X_new = new_data.drop(columns=['CustomerID'])

model = joblib.load("voting_classifier_model.pkl")

scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)  # Fit the scaler to new data

churn_proba = model.predict_proba(X_new_scaled)[:, 1]  # Probability of churn
churn_class = model.predict(X_new_scaled)  # Churn prediction (0 = No, 1 = Yes)

print(f"Churn Probability: {churn_proba[0]:.2f}")
print(f"Churn Prediction (1=Churn, 0=No Churn): {churn_class[0]}")
