from datasets import load_dataset
import pandas as pd
import os

print("Downloading customer reviews...")

dataset = load_dataset(
    "fancyzhx/amazon_polarity",
    split="train",
    streaming=True
)

reviews = []

for i, item in enumerate(dataset):
    if i >= 3000:
        break

    reviews.append({
        "review": item["content"],
        "sentiment": "positive" if item["label"] == 1 else "negative"
    })

df = pd.DataFrame(reviews)

os.makedirs("dataset", exist_ok=True)

df.to_csv("dataset/customer_reviews.csv", index=False)

print("Dataset downloaded successfully!")
print("Total reviews:", len(df))
print(df.head())
