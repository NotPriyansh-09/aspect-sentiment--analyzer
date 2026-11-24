# Aspect Sentiment Analyzer 🧠

A Streamlit web app for multi-domain review sentiment analysis, adapted from the supplied machine-learning notebook.

## Features

- Upload a CSV training dataset
- Train a TF-IDF + multi-output Random Forest classifier
- Predict sentiment across detected domains/aspects
- Review per-domain accuracy and classification reports
- Preview and download prepared training data
- Purple/black gradient interface

## Run locally

Use Python 3.10–3.12.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Then open the local URL Streamlit prints (typically http://localhost:8501).

## Dataset format

Upload a CSV containing these columns:

| Column | Description |
|---|---|
| `Review_Text` | Text of the customer review |
| `Domain` | Aspect/domain label |
| `Sentiment_Label` | `positive`, `negative`, or `neutral` |

The dataset is uploaded through the app; it is not included in this repository.

## Deploy publicly with Streamlit Community Cloud

1. Visit https://share.streamlit.io/
2. Sign in with GitHub and select this repository.
3. Choose branch `main` and app file `app.py`.
4. Deploy. Streamlit will provide a public `*.streamlit.app` URL.

## Note

This is an academic prototype. Evaluation scores and predictions depend on the dataset's size, label quality, and domain coverage.
