
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("dataset/final_customer_analysis.csv")

df = load_data()

# Load AI models
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


# Dashboard heading
st.title("📊 Customer Sentiment & Intent Platform")
st.write("AI-powered customer feedback analytics")
st.divider()

# Calculate statistics
total_reviews = len(df)
positive = (df["predicted_sentiment"] == "POSITIVE").sum()
negative = (df["predicted_sentiment"] == "NEGATIVE").sum()

# KPI cards
col1, col2, col3 = st.columns(3)

col1.metric("Total Reviews", total_reviews)
col2.metric("Positive Reviews", positive)
col3.metric("Negative Reviews", negative)

st.divider()

# Sentiment chart
st.subheader("Customer Sentiment Distribution")

sentiment_counts = (
    df["predicted_sentiment"]
    .value_counts()
    .rename_axis("Sentiment")
    .reset_index(name="Count")
)

fig = px.pie(
    sentiment_counts,
    names="Sentiment",
    values="Count",
    title="Positive vs Negative Reviews",
    hole=0.4
)

st.plotly_chart(fig, width="stretch")

# Sentiment bar chart
st.subheader("Sentiment Comparison")

bar_fig = px.bar(
    sentiment_counts,
    x="Sentiment",
    y="Count",
    color="Sentiment",
    title="Number of Customer Reviews"
)

st.plotly_chart(bar_fig, width="stretch")
# Intent Distribution Chart
st.divider()
# Intent Summary
total_complaints = (df["predicted_intent"] == "Complaint").sum()
total_refunds = (df["predicted_intent"] == "Refund request").sum()
most_common_intent = df["predicted_intent"].mode()[0]

st.subheader("Customer Intent Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Total Complaints", total_complaints)
col2.metric("Refund Requests", total_refunds)
col3.metric("Most Common Intent", most_common_intent)
st.subheader("Customer Intent Distribution")

intent_counts = (
    df["predicted_intent"]
    .value_counts()
    .reset_index()
)

intent_counts.columns = ["Intent", "Count"]

intent_fig = px.bar(
    intent_counts,
    x="Intent",
    y="Count",
    color="Intent",
    title="Customer Intent Categories"
)

st.plotly_chart(intent_fig, width="stretch")

# Customer feedback prediction
st.divider()
st.subheader("Analyze New Customer Feedback")

customer_text = st.text_area(
    "Enter customer feedback",
    placeholder="Example: I received a damaged product and want a refund."
)

if st.button("Analyze Feedback", type="primary"):

    if customer_text.strip():

        with st.spinner("Analyzing customer feedback..."):

            sentiment_model, intent_model = load_models()

            sentiment_result = sentiment_model(
                customer_text,
                truncation=True,
                max_length=512
            )[0]

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

            intent_result = intent_model(
                customer_text,
                candidate_labels=categories
            )

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Sentiment",
            sentiment_result["label"]
        )

        col1.write(
            f"Confidence: {sentiment_result['score']*100:.2f}%"
        )

        col2.metric(
            "Predicted Intent",
            intent_result["labels"][0]
        )

        col2.write(
            f"Confidence: {intent_result['scores'][0]*100:.2f}%"
        )

    else:
        st.warning("Please enter customer feedback.")

# Display dataset
st.divider()
st.subheader("Customer Review Records")

st.dataframe(
    df[["review", "sentiment", "predicted_sentiment", "confidence"]],
    width="stretch"
)

# Download report
csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Customer Analysis Report",
    data=csv_data,
    file_name="customer_analysis_report.csv",
    mime="text/csv"
)

st.divider()

st.caption("Developed using Python, PyTorch, Hugging Face and Streamlit.")
