
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Loading BERT tokenizer...")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Tokenizer loaded successfully!")

print("\nLoading BERT sentiment model...")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name
)

print("BERT model loaded successfully!")

print("\nModel details:")
print(model.config.id2label)

import torch

# Function to predict sentiment
def predict_sentiment(text):

    # Convert text into tokens
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculate probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)

    # Get predicted class
    prediction = torch.argmax(probabilities, dim=1).item()

    sentiment = model.config.id2label[prediction]
    confidence = probabilities[0][prediction].item() * 100

    return sentiment, confidence


# Test customer reviews
reviews = [
    "This product is amazing and I absolutely love it!",
    "The product is terrible and I am very disappointed.",
    "The quality is excellent and delivery was fast."
]

print("\nCUSTOMER SENTIMENT PREDICTIONS")

for review in reviews:
    sentiment, confidence = predict_sentiment(review)

    print("\nReview:", review)
    print("Sentiment:", sentiment)
    print("Confidence:", round(confidence, 2), "%")
