# This script trains a model to predict NIL Tier using football recruit info

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
csv_path = "data/nil_dataset_diverse.csv"
nil_dataframe = pd.read_csv(csv_path)

# Drop rows with missing key values
required_columns = ["Star Rating", "position", "Estimated NIL Budget"]
nil_dataframe = nil_dataframe.dropna(subset=required_columns)

# Encode position as numbers
position_encoder = LabelEncoder()
nil_dataframe["position_encoded"] = position_encoder.fit_transform(nil_dataframe["position"])

# Scale NIL Budget
scaler = StandardScaler()
nil_dataframe["Estimated NIL Budget (scaled)"] = scaler.fit_transform(
    nil_dataframe[["Estimated NIL Budget"]]
)

# Add feature: star rating × scaled budget
nil_dataframe["rating_times_budget"] = (
    nil_dataframe["Star Rating"] * nil_dataframe["Estimated NIL Budget (scaled)"]
)

# Label tiers
def determine_nil_tier(budget):
    if budget >= 10_000_000:
        return "A"
    elif budget >= 5_000_000:
        return "B"
    else:
        return "C"

nil_dataframe["NIL_Tier"] = nil_dataframe["Estimated NIL Budget"].apply(determine_nil_tier)

# Encode target label
tier_encoder = LabelEncoder()
nil_dataframe["NIL_Tier_encoded"] = tier_encoder.fit_transform(nil_dataframe["NIL_Tier"])

# Features and labels
features_to_use = [
    "Star Rating", "position_encoded", "Estimated NIL Budget (scaled)", "rating_times_budget"
]
X = nil_dataframe[features_to_use]
y = nil_dataframe["NIL_Tier_encoded"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

print("=== Random Forest Results ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

# Cross-validation
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("Random Forest Cross-Validation Scores:")
for i, score in enumerate(rf_cv_scores):
    print(f"Fold {i + 1}: {score:.3f}")
print("Average Accuracy:", round(rf_cv_scores.mean(), 3))

# Save model and scaler
os.makedirs("model", exist_ok=True)
joblib.dump(rf_model, "model/random_forest_model.pkl")
joblib.dump(scaler, "model/budget_scaler.pkl")
print("Model and scaler saved.")

# Optional: Logistic Regression comparison
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_predictions = log_model.predict(X_test)

print("\n=== Logistic Regression Results ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, log_predictions))
print("\nClassification Report:")
print(classification_report(y_test, log_predictions))

# Logistic Regression CV
log_cv_scores = cross_val_score(log_model, X, y, cv=5)
print("Logistic Regression Cross-Validation Scores:")
for i, score in enumerate(log_cv_scores):
    print(f"Fold {i + 1}: {score:.3f}")
print("Average Accuracy:", round(log_cv_scores.mean(), 3))

# Plot feature importance
os.makedirs("output", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.barh(features_to_use, rf_model.feature_importances_, color="skyblue", edgecolor="black")
plt.title("Feature Importance: Predicting NIL Tier")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("output/feature_importance.png")
plt.show()