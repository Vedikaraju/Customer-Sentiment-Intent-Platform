
import pandas as pd

# Load our customer reviews
df = pd.read_csv("dataset/customer_reviews.csv")

# Display dataset information
print("Dataset Shape:", df.shape)

print("\nFirst 5 Reviews:")
print(df.head())

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())
