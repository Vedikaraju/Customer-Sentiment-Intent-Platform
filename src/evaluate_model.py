
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load analyzed dataset
df = pd.read_csv("dataset/analyzed_reviews.csv")

# Actual and predicted labels
actual = df["sentiment"].str.upper()
predicted = df["predicted_sentiment"].str.upper()

# Calculate accuracy
accuracy = accuracy_score(actual, predicted)

print("MODEL EVALUATION RESULTS")
print("------------------------")

print("Total Reviews:", len(df))
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(actual, predicted))

print("\nConfusion Matrix:")
print(confusion_matrix(
    actual,
    predicted,
    labels=["NEGATIVE", "POSITIVE"]
))
