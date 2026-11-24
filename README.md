# Aspect Sentiment & Research Studio 🧠

A Streamlit academic project combining aspect-level customer-review sentiment analysis with live web search and searchable PDF notes.

## Features

- **Web research:** Search public web indexes, select India/worldwide scope, view source snippets and links, and export a research brief as PDF.
- **Research digest:** Creates a short extractive digest from search-result snippets. It does not use a generative AI model, and the user should open sources to verify claims.
- **PDF library:** Upload multiple text-based PDFs and search for terms across their pages.
- **Sentiment analysis:** Train a TF-IDF + multi-output Random Forest classifier and predict sentiment across domains/aspects.
- **Evaluation:** Review per-domain accuracy and classification reports.
- **Interactive workspace:** Session prediction history, reset controls, animated purple waves, and optional user-enabled ambient tone.

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

The dataset is uploaded through the app; it is not included in this repository. Web research and PDF search can be used without uploading the CSV.

## Web search and research notes

The app uses the `ddgs` Python library to query public search services without requiring a paid API key. Availability and result quality depend on those external providers; requests may be temporarily rate-limited. The app shows source links and snippets so findings can be checked.

The current research digest is extractive (selected from snippets), not an LLM-written report or a substitute for reading the original sources.

## PDF notes

The PDF reader extracts selectable text. Scanned/image-only PDFs need OCR and may not return searchable text in this version.

## Ambient sound

The optional low-volume tone is off by default. The user must press its button to enable audio because browsers commonly block autoplay.

## Deploy publicly with Streamlit Community Cloud

1. Visit https://share.streamlit.io/
2. Sign in with GitHub and select this repository.
3. Choose branch `main` and app file `app.py`.
4. Deploy. Streamlit will provide a public `*.streamlit.app` URL.

## Note

This is an academic prototype. Evaluation scores and predictions depend on the dataset's size, label quality, and domain coverage.
