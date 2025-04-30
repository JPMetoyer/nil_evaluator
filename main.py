# this is the main script

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


csv_path = "data/nil_dataset.csv"
nil_dataframe = pd.read_csv(csv_path)

columns_required = ["Star Rating", "position", "Estimated NIL Budget"]
nil_dataframe = nil_dataframe.dropna(subset=columns_required)


position_label_encoder = LabelEncoder()
position_label_encoder.fit(nil_dataframe["position"])
nil_dataframe["position_encoded"] = position_label_encoder.transform(nil_dataframe["position"])


def determine_nil_tier(budget_amount):
    if budget_amount >= 10000000:
        return "A"
    elif budget_amount >= 5000000:
        return "B"
    else:
        return "C"

nil_dataframe["NIL_Tier"] = nil_dataframe["Estimated NIL Budget"].apply(determine_nil_tier)

tier_label_encoder = LabelEncoder()
tier_label_encoder.fit(nil_dataframe["NIL_Tier"])
nil_dataframe["NIL_Tier_encoded"] = tier_label_encoder.transform(nil_dataframe["NIL_Tier"])


features_to_use = ["Star Rating", "position_encoded", "Estimated NIL Budget"]
X_features = nil_dataframe[features_to_use]
y_target = nil_dataframe["NIL_Tier_encoded"]


test_fraction = 0.2
split_seed = 42

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_target, test_size=test_fraction, random_state=split_seed
)

number_of_estimators = 100
model_seed_value = 42

rf_model = RandomForestClassifier(
    n_estimators=number_of_estimators,
    random_state=model_seed_value
)


rf_model.fit(X_train, y_train)


test_predictions = rf_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(classification_report(y_test, test_predictions))


fold_count = 5
validation_scores = cross_val_score(estimator=rf_model, X=X_features, y=y_target, cv=fold_count)

print("Cross-Validation Scores by Fold:")
for fold_index, score in enumerate(validation_scores):
    print(f"Fold {fold_index + 1}: {score:.3f}")

average_validation = sum(validation_scores) / len(validation_scores)
print("Average Cross-Validation Accuracy:", round(average_validation, 3))


importance_scores = rf_model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(features_to_use, importance_scores, color="skyblue", edgecolor="black")
plt.title("Feature Importance: Predicting NIL Tier")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


model_output_path = "model/random_forest_model.pkl"
joblib.dump(rf_model, model_output_path)
print(f"Model saved at: {model_output_path}")