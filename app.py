import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once
nltk.download('stopwords')
nltk.download('wordnet')

# Load model
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# 🔥 Page Config
st.set_page_config(page_title="Fake News Detection", layout="centered")

# 🔥 Custom CSS (for same UI look)
st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: white;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .warning-box {
        background-color: #1f3b4d;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #d1ecf1;
    }
    .stTextArea textarea {
        background-color: #1e1e1e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# 🔥 Title
st.markdown('<div class="title">📰 Fake News Detection System</div>', unsafe_allow_html=True)

# 🔥 Warning Box
st.markdown("""
<div class="warning-box">
⚠️ This model was trained on historical political news datasets (ISOT & LIAR).
Predictions on very recent news, non-political topics, or region-specific stories may be less reliable.
Future versions will include live news API training.
</div>
""", unsafe_allow_html=True)

# Subtitle
st.write("Enter a news article below to classify it as Real or Fake.")

# Input
text = st.text_area("News Text")

# Prediction Button
if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(text)
        vector = tfidf.transform([cleaned])
        
        pred = model.predict(vector)[0]
        proba = model.predict_proba(vector)
        
        fake_prob = proba[0][0]
        real_prob = proba[0][1]

        if pred == 1:
            st.success(f"✅ Real News\nConfidence: {real_prob:.4f}")
        else:
            st.error(f"❌ Fake News\nConfidence: {fake_prob:.4f}")