import re
import numpy as np
import pandas as pd
import streamlit as st
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Aspect Sentiment Analyzer", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 10% 5%, #241044 0, #100d20 35%, #080812 100%); color:#f7f3ff;}
[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:2rem; max-width:1200px;}
.hero {padding:28px 30px; border:1px solid #583c83; border-radius:22px;
background:linear-gradient(120deg,rgba(101,45,170,.34),rgba(20,17,40,.88)); margin-bottom:22px;}
.hero h1 {font-size:2.35rem; margin:0; color:#fff;}
.hero p {color:#c9b8e9; margin:.5rem 0 0;}
div[data-testid="stMetric"] {background:#171329; border:1px solid #40305e; padding:16px; border-radius:16px;}
.stButton>button {background:linear-gradient(90deg,#7c3aed,#a855f7); color:white; border:0; border-radius:10px; font-weight:700;}
.stButton>button:hover {border:1px solid #d8b4fe; color:white;}
[data-testid="stFileUploader"] {background:#171329; padding:12px; border:1px dashed #7655a5; border-radius:14px;}
.result-card {background:#171329; border:1px solid #40305e; padding:14px 18px; border-radius:14px; margin:8px 0;}
.small-muted {color:#b9a9d5;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🧠 Aspect Sentiment Analyzer</h1>
  <p>Multi-domain review analysis powered by TF-IDF and a multi-output Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None

@st.cache_resource
def get_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))

def preprocess(text, nlp, stops):
    text = re.sub(r"[^a-zA-Z\\s]", "", str(text).lower())
    if nlp is not None:
        doc = nlp(text)
        words = [t.lemma_ for t in doc if t.pos_ in ("NOUN", "ADJ") and t.text not in stops]
        if words:
            return " ".join(words)
    return " ".join(w for w in text.split() if w not in stops)

def build_training_table(raw):
    required = {"Review_Text", "Domain", "Sentiment_Label"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))
    data = raw.dropna(subset=["Review_Text", "Domain", "Sentiment_Label"]).copy()
    data["Domain"] = data["Domain"].astype(str).str.strip().str.lower()
    data["Sentiment_Label"] = data["Sentiment_Label"].astype(str).str.strip().str.lower()
    data["Review_Text"] = data["Review_Text"].astype(str)
    sentiment_map = {"positive": 1, "negative": 2, "neutral": 0}
    unknown = sorted(set(data["Sentiment_Label"]) - set(sentiment_map))
    if unknown:
        raise ValueError("Unsupported sentiment labels: " + ", ".join(unknown) +
                         ". Use positive, negative, or neutral.")
    aspects = sorted(data["Domain"].unique().tolist())
    grouped = data.groupby("Review_Text", sort=False)
    rows = []
    for review, group in grouped:
        row = {"review": review}
        for aspect in aspects:
            row[aspect] = 0
        for _, item in group.iterrows():
            row[item["Domain"]] = sentiment_map[item["Sentiment_Label"]]
        rows.append(row)
    return pd.DataFrame(rows), aspects, data

def sentiment_name(value):
    return {0: "Neutral / Not mentioned", 1: "Positive", 2: "Negative"}.get(int(value), "Unknown")

with st.sidebar:
    st.header("⚙️ Project controls")
    st.write("Upload the dataset used by your notebook.")
    st.caption("Required columns: Review_Text, Domain, Sentiment_Label")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.divider()
    st.caption("Model: TF-IDF (max 1,500 features) + MultiOutput Random Forest (100 trees).")

if uploaded is None:
    st.info("👈 Upload **Cross_Domain_Dataset.csv** in the sidebar to train the model and unlock predictions.")
    st.markdown("### What this app does")
    c1,c2,c3 = st.columns(3)
    c1.markdown("**01 · Prepare**\\n\\nGroups review rows by review text and maps each domain to a sentiment label.")
    c2.markdown("**02 · Train**\\n\\nExtracts text features with TF-IDF and trains one Random Forest classifier per domain.")
    c3.markdown("**03 · Predict**\\n\\nEnter a review to see predicted sentiment for every domain.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
    table, aspects, clean_raw = build_training_table(raw)
except Exception as e:
    st.error(f"Could not prepare this dataset: {e}")
    st.stop()

if len(table) < 4:
    st.error("The dataset needs at least 4 unique reviews to create a train/test split.")
    st.stop()

st.success(f"Dataset loaded: {len(clean_raw):,} aspect records · {len(table):,} unique reviews · {len(aspects)} domains")
with st.expander("Preview uploaded data"):
    st.dataframe(raw.head(10), use_container_width=True)
    st.write("Detected domains:", ", ".join(aspects))

nlp = load_nlp()
st.caption("Text preprocessing: " + ("spaCy noun/adjective lemmatization" if nlp else "fallback word cleanup (spaCy English model not installed)"))
stops = get_stopwords()

@st.cache_resource(show_spinner="Training the model…")
def train_model(table_records, aspect_tuple):
    frame = pd.DataFrame(table_records)
    aspects_local = list(aspect_tuple)
    frame["cleaned_review"] = frame["review"].apply(lambda x: preprocess(x, load_nlp(), get_stopwords()))
    vectorizer_local = TfidfVectorizer(max_features=1500)
    X = vectorizer_local.fit_transform(frame["cleaned_review"])
    Y = frame[aspects_local].astype(int)
    if len(frame) < 8:
        test_size = max(1, int(round(len(frame)*0.25)))
    else:
        test_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )
    classifier = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        n_jobs=-1
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracies = {a: accuracy_score(y_test[a], predictions[:, i]) for i,a in enumerate(aspects_local)}
    return vectorizer_local, classifier, accuracies, frame, X_test, y_test, predictions

try:
    records = table.to_dict(orient="records")
    vectorizer, model, accuracies, processed, X_test, y_test, y_pred = train_model(records, tuple(aspects))
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

avg_acc = float(np.mean(list(accuracies.values()))) if accuracies else 0.0
m1,m2,m3 = st.columns(3)
m1.metric("Unique reviews", f"{len(table):,}")
m2.metric("Domains / aspects", len(aspects))
m3.metric("Mean test accuracy", f"{avg_acc*100:.1f}%")

tab_predict, tab_metrics, tab_data = st.tabs(["🔍 Analyze a review", "📊 Model evaluation", "🗂️ Dataset"])

with tab_predict:
    st.subheader("Try a review")
    examples = [
        "The laptop performance is incredible, but the battery life is disappointing.",
        "The food was delicious and the staff were very friendly.",
        "The delivery was late and the packaging was damaged."
    ]
    chosen = st.selectbox("Example review (optional)", ["Write my own"] + examples)
    default_text = "" if chosen == "Write my own" else chosen
    review_text = st.text_area("Enter customer review", value=default_text, height=120,
                               placeholder="Type or paste a review here…")
    if st.button("✨ Analyze sentiment", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("Please enter a review first.")
        else:
            cleaned = preprocess(review_text, nlp, stops)
            vector = vectorizer.transform([cleaned])
            predictions = model.predict(vector)[0]
            probabilities = model.predict_proba(vector)
            st.markdown("### Prediction results")
            st.caption("Sentiment is predicted separately for each domain in the uploaded training data.")
            cols = st.columns(2)
            for i, (aspect, value) in enumerate(zip(aspects, predictions)):
                classes = list(model.estimators_[i].classes_)
                probs = probabilities[i][0]
                confidence = float(probs[classes.index(value)]) if value in classes else 0.0
                with cols[i % 2]:
                    st.markdown(f'<div class="result-card"><b>{aspect.title()}</b><br><span>{sentiment_name(value)}</span><br><small class="small-muted">Model confidence: {confidence*100:.1f}%</small></div>', unsafe_allow_html=True)
            with st.expander("See text preprocessing"):
                st.write(cleaned if cleaned else "No noun/adjective terms detected; fallback cleanup produced an empty string.")

with tab_metrics:
    st.subheader("Test-set accuracy by domain")
    metric_df = pd.DataFrame({"Domain": list(accuracies.keys()),
                              "Accuracy": [round(v*100, 2) for v in accuracies.values()]})
    st.bar_chart(metric_df.set_index("Domain"))
    st.dataframe(metric_df, use_container_width=True, hide_index=True)
    with st.expander("Detailed classification report"):
        for i, aspect in enumerate(aspects):
            labels = sorted(set(y_test[aspect].tolist()) | set(y_pred[:, i].tolist()))
            names = [sentiment_name(x) for x in labels]
            report = classification_report(y_test[aspect], y_pred[:, i], labels=labels,
                                           target_names=names, zero_division=0)
            st.markdown(f"**{aspect.title()}**")
            st.code(report)

with tab_data:
    st.subheader("Prepared training data")
    st.dataframe(processed[["review"] + aspects + ["cleaned_review"]].head(100),
                 use_container_width=True)
    st.download_button("⬇️ Download prepared dataset",
                       table.to_csv(index=False).encode("utf-8"),
                       file_name="prepared_aspect_sentiment_dataset.csv",
                       mime="text/csv")

st.markdown("---")
st.caption("Academic prototype. Predictions depend on the quality and coverage of the uploaded dataset.")
