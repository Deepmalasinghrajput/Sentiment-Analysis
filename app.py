import streamlit as st
import joblib

# Load Model & Vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Emoji mapping
emoji_map = {
    "Positive": "😊",
    "Negative": "😞",
    "Neutral": "😐"
}

# Page settings
st.set_page_config(page_title="Text Classifier ✨", layout="centered")

# Center CSS
center_style = """
<style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
    }
    textarea {
        text-align: center;
        font-size: 16px;
    }
</style>
"""
st.markdown(center_style, unsafe_allow_html=True)

# Title
st.markdown("<h2 class='center'> Sentiment Analysis</h2>", unsafe_allow_html=True)

# Input label
st.markdown("<p class='center'> Enter your text here✨</p>", unsafe_allow_html=True)

# Input box
user_input = st.text_area("", height=130, placeholder="Type something here... 😊")

# Predict Button
if st.button(" Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        X = vectorizer.transform([user_input])
        pred = model.predict(X)[0]

        emoji = emoji_map.get(pred, "🤔")  # default emoji if unknown class

        st.success(f"✅ Result: **{pred}** {emoji}")
