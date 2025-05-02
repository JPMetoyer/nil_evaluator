import joblib
import pandas as pd

# load trained model and scaler
model_path = "model/random_forest_model.pkl"
scaler_path = "model/budget_scaler.pkl"
rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# getting user input
star_rating = float(input("Enter star rating (1 to 5)(note: this showcases the overall rating of a player): "))
position = input("Enter player position (QB, WR, RB, OL, DL ,LB, S, TE, ATH): ")
budget_raw = float(input("Enter estimated NIL budget for the school ($) (note: only enter numbers no commas or decmials please): "))

# encoding position manually
position_list = ['QB', 'WR', 'RB', 'OL', 'DL', 'LB', 'CB', 'S', 'TE', 'ATH']
position_encoded = position_list.index(position) if position in position_list else 0

# create input dataframe
input_df = pd.DataFrame([{
    "Star Rating": star_rating,
    "position_encoded": position_encoded,
    "Estimated NIL Budget": budget_raw
}])

# usieng real scaler from training
input_df["Estimated NIL Budget (scaled)"] = scaler.transform(input_df[["Estimated NIL Budget"]])

# ddd interaction term
input_df["rating_times_budget"] = input_df["Star Rating"] * input_df["Estimated NIL Budget (scaled)"]

# final feature set
final_features = input_df[[
    "Star Rating", "position_encoded", "Estimated NIL Budget (scaled)", "rating_times_budget"
]]

# -redict and display result
predicted_class = rf_model.predict(final_features)[0]
tier_mapping = {0: "A", 1: "B", 2: "C"}
print(f"\nPredicted NIL Tier: {tier_mapping.get(predicted_class, '?')}")