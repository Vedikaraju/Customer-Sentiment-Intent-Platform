import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load the dataset
df = pd.read_csv("dataset/analyzed_reviews.csv")

# Load the intent classification model
intent_model = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1",
    device=-1
)

# Intent categories
categories = [
    "Complaint",
    "Refund request",
    "Order inquiry",
    "Product inquiry",
    "Technical support",
    "Cancellation",
    "Delivery issue",
    "Appreciation",
    "Other"
]

# Classify customer reviews in batches
intents = []
confidence_scores = []

batch_size = 8

for start in tqdm(range(0, len(df), batch_size)):
    batch = df["review"].iloc[start:start + batch_size].fillna("").tolist()

    results = intent_model(
        batch,
        candidate_labels=categories,
        multi_label=False
    )

    if isinstance(results, dict):
        results = [results]

    for result in results:
        intents.append(result["labels"][0])
        confidence_scores.append(result["scores"][0])

# Add predictions to dataset
df["predicted_intent"] = intents
df["intent_confidence"] = confidence_scores

# Save results
df.to_csv("dataset/final_customer_analysis.csv", index=False)

print("Intent analysis completed successfully!")
print(df["predicted_intent"].value_counts())