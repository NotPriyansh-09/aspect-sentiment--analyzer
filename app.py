import io
import re
import time
import html
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import spacy
import nltk
from nltk.corpus import stopwords
from pypdf import PdfReader
from ddgs import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib import colors

st.set_page_config(page_title="Aspect Sentiment & Research Studio", page_icon="🧠", layout="wide")

# ---------- Visual design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { color-scheme: dark; }
.stApp {
  background: radial-gradient(ellipse at 12% 8%, #32145c 0%, transparent 38%),
              radial-gradient(ellipse at 88% 22%, #21104a 0%, transparent 34%),
              linear-gradient(135deg,#080711 0%,#100b20 48%,#090812 100%);
  color:#f7f1ff; font-family:Inter,sans-serif;
}
[data-testid="stHeader"] {background:rgba(8,7,17,.45);}
.block-container {padding-top:1.6rem;max-width:1320px;}
.hero {
 position:relative; overflow:hidden; padding:30px 32px; border:1px solid rgba(192,132,252,.38);
 border-radius:24px; background:linear-gradient(115deg,rgba(111,45,185,.32),rgba(22,15,43,.82));
 box-shadow:0 15px 55px rgba(96,40,160,.17); margin-bottom:22px;
}
.hero:before,.hero:after {content:"";position:absolute;left:-15%;width:130%;height:95px;border-radius:45%;
 border:2px solid rgba(192,132,252,.18);pointer-events:none;animation:waveDrift 11s ease-in-out infinite alternate;}
.hero:before {bottom:-62px;transform:rotate(-4deg);}
.hero:after {bottom:-78px;border-color:rgba(168,85,247,.2);animation-duration:15s;animation-direction:alternate-reverse;}
@keyframes waveDrift {from{transform:translateX(-3%) rotate(-3deg)}to{transform:translateX(3%) rotate(3deg)}}
.hero h1 {font-size:clamp(1.8rem,4vw,2.7rem);font-weight:800;letter-spacing:-1px;margin:0;color:#fff;}
.hero p {color:#d8c5f3;margin:.55rem 0 0;max-width:760px;}
.glass {background:rgba(24,17,43,.78);border:1px solid rgba(167,139,250,.23);border-radius:17px;padding:16px 18px;}
.result-card {background:linear-gradient(120deg,rgba(55,28,85,.75),rgba(21,17,36,.9));border:1px solid #62418b;padding:15px 18px;border-radius:15px;margin:8px 0;}
.small-muted {color:#c4b5d8;font-size:.88rem;}
div[data-testid="stMetric"] {background:rgba(27,19,46,.82);border:1px solid #4c3470;padding:15px;border-radius:15px;}
.stButton>button,.stDownloadButton>button {background:linear-gradient(95deg,#7c3aed,#a855f7);color:white;border:0;border-radius:11px;font-weight:700;transition:all .2s ease;}
.stButton>button:hover,.stDownloadButton>button:hover {box-shadow:0 0 22px rgba(168,85,247,.42);transform:translateY(-1px);color:white;border:1px solid #d8b4fe;}
[data-testid="stFileUploader"] {background:rgba(24,17,43,.75);padding:12px;border:1px dashed #7655a5;border-radius:14px;}
.stTextInput input,.stTextArea textarea {background:#171329!important;color:#fff!important;border-color:#65458c!important;}
.stTabs [data-baseweb="tab"] {color:#d9c8ef;}
.stTabs [aria-selected="true"] {color:#e9d5ff!important;}
div[data-testid="stExpander"] {background:rgba(24,17,43,.5);border:1px solid #483166;border-radius:13px;}
div[data-testid="stAlert"] {border-radius:14px;}
@media (prefers-reduced-motion: reduce) {*,*:before,*:after {animation:none!important;transition:none!important;}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🧠 Aspect Sentiment & Research Studio</h1>
  <p>Explore customer sentiment, search the live web, investigate uploaded PDFs, and export a source-linked research brief — all in one purple-wave workspace.</p>
</div>
""", unsafe_allow_html=True)

# User-controlled gentle ambient sound. Browsers require a user gesture before audio can play.
with st.expander("🎧 Focus ambience", expanded=False):
    st.caption("Optional, very soft ambient tone. Press play to enable it; your browser may block audio until you interact.")
    components.html("""
    <div style="font-family:Arial;color:#e9d5ff;background:#171329;border:1px solid #62418b;border-radius:12px;padding:12px">
      <button id="amb" style="background:linear-gradient(90deg,#7c3aed,#a855f7);color:white;border:0;border-radius:9px;padding:9px 14px;font-weight:bold;cursor:pointer">▶ Enable soft ambient sound</button>
      <span id="state" style="margin-left:10px;font-size:12px;color:#c4b5d8">Sound is off</span>
    </div>
    <script>
    (() => {
      const b=document.getElementById('amb'),s=document.getElementById('state');
      let ctx=null,osc=null,gain=null,on=false;
      b.addEventListener('click', async()=>{
        try {
          if(!on){
            ctx=new (window.AudioContext||window.webkitAudioContext)();
            await ctx.resume();
            osc=ctx.createOscillator(); gain=ctx.createGain();
            osc.type='sine'; osc.frequency.value=174;
            gain.gain.value=0.012; osc.connect(gain); gain.connect(ctx.destination); osc.start();
            on=true;b.textContent='Ⅱ Pause ambient sound';s.textContent='Soft tone is playing';
          } else {
            if(ctx) await ctx.close();ctx=null;on=false;
            b.textContent='▶ Enable soft ambient sound';s.textContent='Sound is off';
          }
        } catch(e){s.textContent='Audio unavailable in this browser';}
      });
    })();
    </script>
    """, height=72)

# ---------- Shared helpers ----------
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
    text = re.sub(r"[^a-zA-Z\\s]", " ", str(text).lower())
    if nlp is not None:
        doc = nlp(text)
        words = [t.lemma_ for t in doc if t.pos_ in ("NOUN", "ADJ") and t.text not in stops]
        if words:
            return " ".join(words)
    return " ".join(w for w in text.split() if w not in stops)

def sentiment_name(value):
    return {0:"Neutral / Not mentioned",1:"Positive",2:"Negative"}.get(int(value),"Unknown")

def build_training_table(raw):
    required={"Review_Text","Domain","Sentiment_Label"}
    missing=required-set(raw.columns)
    if missing:
        raise ValueError("CSV is missing required columns: "+", ".join(sorted(missing)))
    data=raw.dropna(subset=["Review_Text","Domain","Sentiment_Label"]).copy()
    data["Domain"]=data["Domain"].astype(str).str.strip().str.lower()
    data["Sentiment_Label"]=data["Sentiment_Label"].astype(str).str.strip().str.lower()
    data["Review_Text"]=data["Review_Text"].astype(str)
    mapping={"positive":1,"negative":2,"neutral":0}
    unknown=sorted(set(data["Sentiment_Label"])-set(mapping))
    if unknown:
        raise ValueError("Unsupported sentiment labels: "+", ".join(unknown)+". Use positive, negative, or neutral.")
    aspects=sorted(data["Domain"].unique().tolist())
    rows=[]
    for review,group in data.groupby("Review_Text",sort=False):
        row={"review":review}
        row.update({a:0 for a in aspects})
        for _,item in group.iterrows():
            row[item["Domain"]]=mapping[item["Sentiment_Label"]]
        rows.append(row)
    return pd.DataFrame(rows),aspects,data

def extract_pdfs(files):
    docs=[]
    for f in files or []:
        try:
            reader=PdfReader(io.BytesIO(f.getvalue()))
            pages=[]
            for page_num,page in enumerate(reader.pages,1):
                text=page.extract_text() or ""
                if text.strip():
                    pages.append({"page":page_num,"text":text})
            docs.append({"name":f.name,"pages":pages,"text":"\n".join(p["text"] for p in pages)})
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")
    return docs

def pdf_report_bytes(title, query, findings, sources):
    buffer=io.BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=18*mm,bottomMargin=18*mm)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PurpleTitle",parent=styles["Title"],textColor=colors.HexColor("#6b21a8"),fontSize=20,leading=25))
    styles.add(ParagraphStyle(name="BodySmall",parent=styles["BodyText"],fontSize=9,leading=13,spaceAfter=7))
    story=[Paragraph(html.escape(title),styles["PurpleTitle"]),Spacer(1,8),Paragraph("<b>Research question:</b> "+html.escape(query),styles["BodySmall"]),Spacer(1,8)]
    story.append(Paragraph("Research digest",styles["Heading2"]))
    for item in findings:
        story.append(Paragraph("• "+html.escape(item),styles["BodySmall"]))
    story.append(Spacer(1,8));story.append(Paragraph("Sources",styles["Heading2"]))
    for src in sources:
        line=f'{src.get("title","Source")} — {src.get("url","")}'
        story.append(Paragraph(html.escape(line),styles["BodySmall"]))
    doc.build(story)
    return buffer.getvalue()

def search_web(query, count, region):
    # DDGS is a keyless metasearch library. Results are best-effort and may be rate-limited by upstream engines.
    with DDGS(timeout=15) as ddgs:
        return list(ddgs.text(query, region=region, safesearch="moderate", max_results=count, backend="auto"))

def make_digest(query, results):
    # Transparent extractive synthesis from result snippets; no generative AI model/API is used here.
    snippets=[]
    qterms={w.lower() for w in re.findall(r"[a-zA-Z0-9]+",query) if len(w)>2}
    for r in results:
        text=(r.get("body") or r.get("snippet") or "").strip()
        if text:
            for sentence in re.split(r"(?<=[.!?])\\s+",text):
                sentence=sentence.strip()
                if len(sentence)>35:
                    terms=set(re.findall(r"[a-zA-Z0-9]+",sentence.lower()))
                    score=len(qterms & terms)/(len(qterms) or 1)
                    snippets.append((score,sentence,r.get("title","Web source")))
    snippets.sort(key=lambda x:x[0],reverse=True)
    unique=[]
    seen=set()
    for score,sentence,source in snippets:
        key=sentence.lower()
        if key not in seen:
            unique.append(sentence);seen.add(key)
        if len(unique)>=6: break
    return unique or ["Search results were returned, but the providers did not supply readable text snippets. Open the source links to review the pages."]

# ---------- Sidebar uploads ----------
with st.sidebar:
    st.header("⚙️ Workspace")
    csv_file=st.file_uploader("Training dataset (CSV)",type=["csv"],help="Needs Review_Text, Domain, Sentiment_Label columns.")
    pdf_files=st.file_uploader("Research documents (PDF)",type=["pdf"],accept_multiple_files=True)
    st.divider()
    st.caption("Web research uses public search-engine results. No search API key is required.")
    if st.button("↻ Reset session",use_container_width=True):
        for key in ["prediction_history","research_results","research_digest","last_query","pdf_docs"]:
            st.session_state.pop(key,None)
        st.rerun()

if "prediction_history" not in st.session_state: st.session_state.prediction_history=[]
if "research_results" not in st.session_state: st.session_state.research_results=[]
if "research_digest" not in st.session_state: st.session_state.research_digest=[]
if "last_query" not in st.session_state: st.session_state.last_query=""

pdf_docs=extract_pdfs(pdf_files) if pdf_files else []
st.session_state.pdf_docs=pdf_docs

# ---------- Main navigation ----------
tabs=st.tabs(["🌐 Web research","📄 PDF library","🔍 Sentiment analyzer","📈 Dashboard"])

with tabs[0]:
    st.subheader("Ask a research question")
    st.write("Search the public web, review source snippets, compare findings, and export a brief.")
    with st.form("web_research_form"):
        query=st.text_input("Research topic or question",placeholder="e.g. How is AI being used in predictive maintenance?")
        c1,c2=st.columns([1,1])
        with c1: result_count=st.slider("Number of sources",min_value=3,max_value=10,value=6)
        with c2: region=st.selectbox("Search region",["in-en","wt-wt","us-en"],format_func=lambda x:{"in-en":"India","wt-wt":"Worldwide","us-en":"United States"}[x])
        submitted=st.form_submit_button("✨ Research the web",use_container_width=True)
    if submitted:
        if not query.strip():
            st.warning("Enter a research question first.")
        else:
            status=st.status("Starting your research…",expanded=True)
            progress=st.progress(8)
            status.write("🌊 Preparing search terms and connecting to public search providers…")
            try:
                progress.progress(35)
                status.write("🔎 Searching multiple public web indexes…")
                found=search_web(query.strip(),result_count,region)
                progress.progress(78)
                status.write("🧩 Comparing snippets and assembling a source-linked digest…")
                digest=make_digest(query,found)
                st.session_state.research_results=found
                st.session_state.research_digest=digest
                st.session_state.last_query=query.strip()
                progress.progress(100)
                status.update(label=f"Research ready — {len(found)} source results",state="complete",expanded=False)
            except Exception as e:
                status.update(label="Web search could not be completed",state="error",expanded=True)
                st.error("The search provider may be temporarily unavailable or rate-limiting requests. Try again in a little while.")
                st.caption(f"Technical detail: {str(e)[:240]}")
    if st.session_state.research_results:
        st.markdown("### 🧠 Research digest")
        st.caption("Automatic extractive digest based on returned web snippets. It is not an LLM-generated fact-check; open sources to verify details.")
        for i,item in enumerate(st.session_state.research_digest,1):
            st.markdown(f'<div class="result-card"><b>Finding {i}</b><br>{html.escape(item)}</div>',unsafe_allow_html=True)
        st.markdown("### 🔗 Sources")
        for i,r in enumerate(st.session_state.research_results,1):
            title=r.get("title") or "Untitled source"
            url=r.get("href") or r.get("url") or ""
            body=r.get("body") or r.get("snippet") or "No snippet supplied."
            st.markdown(f"**{i}. [{title}]({url})**")
            st.write(body)
            st.caption(url)
        report=pdf_report_bytes("Web Research Brief",st.session_state.last_query,st.session_state.research_digest,[
            {"title":r.get("title") or "Web source","url":r.get("href") or r.get("url") or ""} for r in st.session_state.research_results
        ])
        st.download_button("📥 Download research brief (PDF)",report,file_name="web_research_brief.pdf",mime="application/pdf")
    else:
        st.markdown('<div class="glass">✨ <b>While you explore:</b> Try a focused question, choose India or worldwide results, then compare the source snippets before using the findings.</div>',unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Search inside uploaded PDFs")
    if not pdf_docs:
        st.info("Upload one or more text-based PDFs in the sidebar to build your searchable document library.")
    else:
        total_pages=sum(len(d["pages"]) for d in pdf_docs)
        st.success(f"{len(pdf_docs)} PDF file(s) loaded · {total_pages} text-readable pages")
        pdf_query=st.text_input("Search all uploaded PDFs",key="pdf_query",placeholder="Enter keywords or a phrase…")
        if pdf_query.strip():
            matches=[]
            for d in pdf_docs:
                for p in d["pages"]:
                    if pdf_query.lower() in p["text"].lower():
                        idx=p["text"].lower().find(pdf_query.lower())
                        snippet=p["text"][max(0,idx-180):idx+len(pdf_query)+220].replace("\n"," ")
                        matches.append((d["name"],p["page"],snippet))
            if matches:
                st.write(f"Found {len(matches)} matching page(s).")
                for name,page,snip in matches[:30]:
                    st.markdown(f'<div class="result-card"><b>{html.escape(name)}</b> · Page {page}<br>{html.escape(snip)}</div>',unsafe_allow_html=True)
            else:
                st.warning("No exact text matches found. Try fewer or different keywords.")
        with st.expander("Document details"):
            for d in pdf_docs:
                st.write(f"**{d['name']}** — {len(d['pages'])} text-readable pages, {len(d['text']):,} extracted characters.")
        st.caption("Scanned PDFs that contain only images need OCR; this version searches selectable/extractable PDF text.")

with tabs[2]:
    st.subheader("Aspect-level sentiment analysis")
    if csv_file is None:
        st.info("Upload your Cross_Domain_Dataset.csv in the sidebar to train the sentiment model. Web research and PDF search work without it.")
    else:
        try:
            raw=pd.read_csv(csv_file)
            table,aspects,clean_raw=build_training_table(raw)
            if len(table)<4:
                st.error("The dataset needs at least 4 unique reviews to create a train/test split.")
            else:
                @st.cache_resource
                def train_model(table_records,aspect_tuple):
                    frame=pd.DataFrame(table_records)
                    aspects_local=list(aspect_tuple)
                    frame["cleaned_review"]=frame["review"].apply(lambda x:preprocess(x,load_nlp(),get_stopwords()))
                    vectorizer_local=TfidfVectorizer(max_features=1500)
                    X=vectorizer_local.fit_transform(frame["cleaned_review"])
                    Y=frame[aspects_local].astype(int)
                    test_size=max(1,int(round(len(frame)*.25))) if len(frame)<8 else .25
                    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=test_size,random_state=42)
                    classifier=MultiOutputClassifier(RandomForestClassifier(n_estimators=100,random_state=42,class_weight="balanced"),n_jobs=-1)
                    classifier.fit(X_train,y_train)
                    predictions=classifier.predict(X_test)
                    accuracies={a:accuracy_score(y_test[a],predictions[:,i]) for i,a in enumerate(aspects_local)}
                    return vectorizer_local,classifier,accuracies,frame,X_test,y_test,predictions
                with st.spinner("Training the sentiment model…"):
                    vectorizer,model,accuracies,processed,X_test,y_test,y_pred=train_model(table.to_dict("records"),tuple(aspects))
                st.success(f"Dataset ready · {len(clean_raw):,} aspect records · {len(table):,} unique reviews · {len(aspects)} domains")
                k1,k2,k3=st.columns(3)
                k1.metric("Unique reviews",f"{len(table):,}")
                k2.metric("Domains",len(aspects))
                k3.metric("Mean test accuracy",f"{np.mean(list(accuracies.values()))*100:.1f}%")
                sample_reviews=["The laptop performance is incredible, but the battery life is disappointing.","The food was delicious and the staff were very friendly.","The delivery was late and the packaging was damaged."]
                sample=st.selectbox("Quick example",["Write my own"]+sample_reviews)
                review=st.text_area("Customer review",value="" if sample=="Write my own" else sample,height=110,key="sentiment_review")
                a,b=st.columns([1,1])
                with a:
                    analyze=st.button("✨ Analyze sentiment",use_container_width=True)
                with b:
                    if st.button("Clear review",use_container_width=True):
                        st.session_state["sentiment_review"]=""
                        st.rerun()
                if analyze:
                    if not review.strip():
                        st.warning("Enter a review first.")
                    else:
                        nlp=load_nlp();stops=get_stopwords()
                        cleaned=preprocess(review,nlp,stops)
                        vector=vectorizer.transform([cleaned])
                        pred=model.predict(vector)[0]
                        probs=model.predict_proba(vector)
                        output=[]
                        cols=st.columns(2)
                        for i,(aspect,value) in enumerate(zip(aspects,pred)):
                            classes=list(model.estimators_[i].classes_)
                            confidence=float(probs[i][0][classes.index(value)]) if value in classes else 0.0
                            output.append({"domain":aspect,"sentiment":sentiment_name(value),"confidence":confidence})
                            with cols[i%2]:
                                st.markdown(f'<div class="result-card"><b>{html.escape(aspect.title())}</b><br>{sentiment_name(value)}<br><span class="small-muted">Confidence: {confidence*100:.1f}%</span></div>',unsafe_allow_html=True)
                        st.session_state.prediction_history.insert(0,{"review":review,"results":output})
                        with st.expander("Preprocessed text"):
                            st.write(cleaned or "No terms remained after preprocessing.")
                with st.expander("Model evaluation"):
                    metric_df=pd.DataFrame({"Domain":list(accuracies.keys()),"Accuracy (%)":[round(v*100,2) for v in accuracies.values()]})
                    st.bar_chart(metric_df.set_index("Domain"))
                    st.dataframe(metric_df,use_container_width=True,hide_index=True)
                    for i,aspect in enumerate(aspects):
                        labels=sorted(set(y_test[aspect].tolist())|set(y_pred[:,i].tolist()))
                        st.markdown(f"**{aspect.title()}**")
                        st.code(classification_report(y_test[aspect],y_pred[:,i],labels=labels,target_names=[sentiment_name(x) for x in labels],zero_division=0))
                with st.expander("Prepared dataset preview"):
                    st.dataframe(processed[["review"]+aspects+["cleaned_review"]].head(100),use_container_width=True)
                    st.download_button("Download prepared CSV",table.to_csv(index=False).encode("utf-8"),"prepared_aspect_sentiment_dataset.csv","text/csv")
        except Exception as e:
            st.error(f"Could not load or train from this CSV: {e}")

with tabs[3]:
    st.subheader("Workspace overview")
    p1,p2,p3=st.columns(3)
    p1.metric("PDF documents",len(pdf_docs))
    p2.metric("PDF text pages",sum(len(d["pages"]) for d in pdf_docs))
    p3.metric("Saved predictions",len(st.session_state.prediction_history))
    st.markdown("### Recent prediction history")
    if st.session_state.prediction_history:
        for item in st.session_state.prediction_history[:10]:
            st.markdown(f'<div class="result-card"><b>{html.escape(item["review"][:180])}</b></div>',unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(item["results"]),use_container_width=True,hide_index=True)
        if st.button("Clear prediction history"):
            st.session_state.prediction_history=[]
            st.rerun()
    else:
        st.caption("Your sentiment analyses will appear here during this session.")
    if st.session_state.research_results:
        st.markdown("### Latest research")
        st.write(f"**Question:** {st.session_state.last_query}")
        st.write(f"**Sources collected:** {len(st.session_state.research_results)}")

st.markdown("---")
st.caption("Academic prototype · Web results are best-effort snippets from third-party search providers. Always open and verify sources before relying on them.")
