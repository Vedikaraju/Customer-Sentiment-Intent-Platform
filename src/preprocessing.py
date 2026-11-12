
import pandas as pd
import re

# Load dataset
df = pd.read_csv("dataset/customer_reviews.csv")

# Function to clean reviews
def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply cleaning
df["cleaned_review"] = df["review"].apply(clean_text)

# Save processed dataset
df.to_csv("dataset/cleaned_reviews.csv", index=False)

print("Data preprocessing completed!")
print("\nOriginal Review:")
print(df["review"][0])

print("\nCleaned Review:")
print(df["cleaned_review"][0])

print("\nProcessed dataset shape:", df.shape)
