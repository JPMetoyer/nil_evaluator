import joblib
import pandas as pd

# Load trained model and scaler
model_path = "model/random_forest_model.pkl"
scaler_path = "model/budget_scaler.pkl"
rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Get user input
star_rating = float(input("Enter star rating (1 to 5): "))
position = input("Enter player position (e.g., QB, WR): ")
budget_raw = float(input("Enter estimated NIL budget for the school ($): "))

# Encode position manually
position_list = ['QB', 'WR', 'RB', 'OL', 'DL', 'LB', 'CB', 'S', 'TE', 'ATH']
position_encoded = position_list.index(position) if position in position_list else 0

# Create input DataFrame
input_df = pd.DataFrame([{
    "Star Rating": star_rating,
    "position_encoded": position_encoded,
    "Estimated NIL Budget": budget_raw
}])

# Use real scaler from training
input_df["Estimated NIL Budget (scaled)"] = scaler.transform(input_df[["Estimated NIL Budget"]])

# Add interaction term
input_df["rating_times_budget"] = input_df["Star Rating"] * input_df["Estimated NIL Budget (scaled)"]

# Final feature set
final_features = input_df[[
    "Star Rating", "position_encoded", "Estimated NIL Budget (scaled)", "rating_times_budget"
]]

# Predict and display result
predicted_class = rf_model.predict(final_features)[0]
tier_mapping = {0: "A", 1: "B", 2: "C"}
print(f"\nPredicted NIL Tier: {tier_mapping.get(predicted_class, '?')}")