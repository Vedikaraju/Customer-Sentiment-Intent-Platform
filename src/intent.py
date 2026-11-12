
from transformers import pipeline

print("Loading customer intent model...")

# Load zero-shot classification model
classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1",
    device=-1
)

print("Intent model loaded successfully!")

# Customer intent categories
intent_categories = [
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

# Function to predict intent
def predict_intent(text):

    result = classifier(
        text,
        candidate_labels=intent_categories
    )

    intent = result["labels"][0]
    confidence = result["scores"][0] * 100

    return intent, confidence


# Test customer messages
test_messages = [
    "I want my money back for this product.",
    "Where is my order?",
    "The product quality is very bad.",
    "I love your service!",
    "How can I reset my password?"
]

print("\nCUSTOMER INTENT PREDICTIONS")

for message in test_messages:

    intent, confidence = predict_intent(message)

    print("\nCustomer Message:", message)
    print("Predicted Intent:", intent)
    print("Confidence:", round(confidence, 2), "%")
