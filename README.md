# Customer Sentiment & Intent Platform

An AI-powered customer feedback analytics platform that analyzes customer reviews using Natural Language Processing (NLP) and Transformer-based models.

## Project Overview

The platform automatically identifies customer sentiment and classifies customer feedback into different intent categories. It provides an interactive dashboard to help understand customer opinions and common issues.

## Features

- Customer review dataset containing 3,000 reviews.
- Text preprocessing and cleaning.
- Sentiment analysis using DistilBERT.
- Intent classification using zero-shot learning.
- Model evaluation using accuracy, precision, recall and F1-score.
- Live customer feedback analysis.
- Interactive Streamlit dashboard.
- Sentiment and intent distribution charts.
- Customer analysis report download.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- DistilBART
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Plotly

## Project Structure

```text
Customer-Sentiment-Intent-Platform/
│
├── dataset/
│   ├── customer_reviews.csv
│   ├── cleaned_reviews.csv
│   ├── analyzed_reviews.csv
│   └── final_customer_analysis.csv
│
├── models/
├── src/
│   ├── download_dataset.py
│   ├── explore_dataset.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── analyze_reviews.py
│   ├── evaluate_model.py
│   ├── intent.py
│   └── analyze_intents.py
│
├── app.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

4. Run the analytics dashboard:

```bash
streamlit run dashboard.py
```

## Machine Learning Models

- **Sentiment Analysis:** distilbert-base-uncased-finetuned-sst-2-english
- **Intent Classification:** valhalla/distilbart-mnli-12-1

## Dataset

Amazon Polarity customer review dataset, using a sample of 3,000 reviews.

## Author

Vedika Kadimella