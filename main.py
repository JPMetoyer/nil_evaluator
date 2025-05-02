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

# loading dataset
csv_path = "data/nil_dataset_diverse.csv"
nil_dataframe = pd.read_csv(csv_path)

# drop rows with missing values in our indicated important columns
required_columns = ["Star Rating", "position", "Estimated NIL Budget"]
nil_dataframe = nil_dataframe.dropna(subset=required_columns)

# encoding positinos as numbesr so that they can be processed by the model
position_encoder = LabelEncoder()
nil_dataframe["position_encoded"] = position_encoder.fit_transform(nil_dataframe["position"])

# scaling the budget to make it easier for the model to learn
scaler = StandardScaler()
nil_dataframe["Estimated NIL Budget (scaled)"] = scaler.fit_transform(nil_dataframe[["Estimated NIL Budget"]]
)

# adding this feature to the dataset bc it was leaning too hard on the nil budge
nil_dataframe["rating_times_budget"] = (
    nil_dataframe["Star Rating"] * nil_dataframe["Estimated NIL Budget (scaled)"]
)

# labeling the tires based on that schools budget
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

# features and labels
features_to_use = [
    "Star Rating", "position_encoded", "Estimated NIL Budget (scaled)", "rating_times_budget"
]
X = nil_dataframe[features_to_use]
y = nil_dataframe["NIL_Tier_encoded"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

print("=== Random Forest Results ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

# cross-validation
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("Random Forest Cross-Validation Scores:")
for i, score in enumerate(rf_cv_scores):
    print(f"Fold {i + 1}: {score:.3f}")
print("Average Accuracy:", round(rf_cv_scores.mean(), 3))

# saving model and sscaler
os.makedirs("model", exist_ok=True)
joblib.dump(rf_model, "model/random_forest_model.pkl")
joblib.dump(scaler, "model/budget_scaler.pkl")
print("if you see this the model and scaler were saved successfully")

# logistic regression comparisons to see if it performs better or worse
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_predictions = log_model.predict(X_test)

print("\n___Logistic Regression Results___")
print("Confusion Matrix:")
print(confusion_matrix(y_test, log_predictions))
print("\nClassification Report:")
print(classification_report(y_test, log_predictions))

# logistic regression to cross validate the scores
log_cv_scores = cross_val_score(log_model, X, y, cv=5)
print("Logistic Regression Cross-Validation Scores:")
for i, score in enumerate(log_cv_scores):
    print(f"Fold {i + 1}: {score:.3f}")
print("Average Accuracy:", round(log_cv_scores.mean(), 3))

# plot feature importance
os.makedirs("output", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.barh(features_to_use, rf_model.feature_importances_, color="skyblue", edgecolor="black")
plt.title("Feature Importance: Predicting NIL Tier")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("output/feature_importance.png")
plt.show()