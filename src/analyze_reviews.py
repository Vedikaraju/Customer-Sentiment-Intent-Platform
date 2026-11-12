
import pandas as pd
from transformers import pipeline

print("Loading sentiment model...", flush=True)

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

print("Model loaded successfully!", flush=True)

df = pd.read_csv("dataset/cleaned_reviews.csv")

predictions = []

# Process reviews in batches of 8
batch_size = 8

for i in range(0, len(df), batch_size):

    batch = df["cleaned_review"].iloc[i:i+batch_size].fillna("").tolist()

    results = classifier(
        batch,
        truncation=True,
        max_length=256
    )

    predictions.extend(results)

    print(
        f"Processed {len(predictions)}/{len(df)} reviews",
        flush=True
    )

df["predicted_sentiment"] = [
    result["label"] for result in predictions
]

df["confidence"] = [
    round(result["score"] * 100, 2)
    for result in predictions
]

df.to_csv("dataset/analyzed_reviews.csv", index=False)

print("\nAnalysis completed successfully!")
print(df["predicted_sentiment"].value_counts())

