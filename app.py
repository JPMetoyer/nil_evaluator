# predict.py
# Run NIL Tier predictions directly from terminal input

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load saved model
model_path = "model/random_forest_model.pkl"
model = joblib.load(model_path)

# Manually rebuild position encoder (same order as training)
position_labels = ['QB', 'WR', 'RB', 'LB', 'DL', 'OL', 'DB', 'TE', 'K', 'ATH']
position_encoder = LabelEncoder()
position_encoder.fit(position_labels)

# Welcome message
print("Welcome to the NIL Tier Predictor!\n")

try:
    # Gather input from user
    star_input = int(input("Enter star rating (1 to 5): "))
    position_input = input("Enter player position (e.g., QB, WR): ").upper()
    budget_input = float(input("Enter estimated NIL budget for the school ($): "))

    # Validate position
    if position_input not in position_labels:
        raise ValueError("Invalid position entered.")

    position_encoded = position_encoder.transform([position_input])[0]

    # Create DataFrame for prediction
    user_data = pd.DataFrame({
        "Star Rating": [star_input],
        "position_encoded": [position_encoded],
        "Estimated NIL Budget": [budget_input]
    })

    # Predict NIL Tier
    prediction = model.predict(user_data)[0]
    decoded_tiers = {0: "A", 1: "B", 2: "C"}
    predicted_tier = decoded_tiers.get(prediction, "Unknown")

    # Output result
    print(f"\nPredicted NIL Tier: {predicted_tier}")

except ValueError as ve:
    print(f"\nInput error: {ve}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")