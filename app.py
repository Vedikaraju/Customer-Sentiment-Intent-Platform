
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Customer Sentiment & Intent Platform",
    page_icon="💬",
    layout="wide"
)

# Load models only once
@st.cache_resource
def load_models():

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

    intent_model = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1",
        device=-1
    )

    return sentiment_model, intent_model


st.title("💬 Customer Sentiment & Intent Platform")

st.write(
    "Analyze customer feedback using AI and "
    "understand customer sentiment and intent."
)

st.divider()

# Customer intent descriptions
intent_categories = [
    "a complaint about poor product or service",
    "a request for a refund or money back",
    "a question about an order or its status",
    "a question about product features or availability",
    "a request for technical support or password reset",
    "a request to cancel an order or subscription",
    "a complaint or question about delivery",
    "positive appreciation or praise",
    "other customer feedback"
]

intent_names = {
    intent_categories[0]: "Complaint",
    intent_categories[1]: "Refund Request",
    intent_categories[2]: "Order Inquiry",
    intent_categories[3]: "Product Inquiry",
    intent_categories[4]: "Technical Support",
    intent_categories[5]: "Cancellation",
    intent_categories[6]: "Delivery Issue",
    intent_categories[7]: "Appreciation",
    intent_categories[8]: "Other"
}

# Input section
st.subheader("Enter Customer Feedback")

customer_text = st.text_area(
    "Customer Review",
    placeholder="Example: I received a damaged product and want a refund.",
    height=120
)

if st.button("Analyze Feedback", type="primary"):

    if customer_text.strip():

        with st.spinner("Loading AI models and analyzing feedback..."):

            sentiment_model, intent_model = load_models()

            sentiment_result = sentiment_model(
                customer_text,
                truncation=True,
                max_length=512
            )[0]

            intent_result = intent_model(
                customer_text,
                candidate_labels=intent_categories,
                multi_label=False
            )

        sentiment = sentiment_result["label"]
        sentiment_score = sentiment_result["score"] * 100

        intent = intent_names[intent_result["labels"][0]]
        intent_score = intent_result["scores"][0] * 100

        st.success("Analysis completed successfully!")

        st.subheader("AI Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Customer Sentiment", sentiment)
            st.progress(sentiment_score / 100)
            st.write(f"Confidence: {sentiment_score:.2f}%")

        with col2:
            st.metric("Customer Intent", intent)
            st.progress(intent_score / 100)
            st.write(f"Confidence: {intent_score:.2f}%")

    else:
        st.warning("Please enter customer feedback first.")

st.divider()

st.caption(
    "Powered by Python, PyTorch, Hugging Face and DistilBERT."
)
