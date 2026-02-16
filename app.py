import streamlit as st
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords

# load nltk
def load_Stopwords():
    nltk.download('stopwords')
    return set(stopwords.words('english'))

# Text preprocessing
def Clean_Data(text):
    stop_words=load_Stopwords()
    text = re.sub(r'[^\w\s]', "", text)
    token=[t.lower() for t in text.split() if t.lower() not in stop_words]
    return " ".join(token)

def detect_new_text_topic(new_text, top_n=10):
    new_text_clean = Clean_Data(new_text)
    lda = joblib.load("lda_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    new_x = vectorizer.transform([new_text_clean])
    topic_distribution = lda.transform(new_x)
    best_index = topic_distribution.argmax()

    feature_names = vectorizer.get_feature_names_out()
    topic = lda.components_[best_index]
    top_indices = topic.argsort()[-top_n:][::-1]
    top_words = [feature_names[i] for i in top_indices]

    topic_name = "-".join(top_words[:3])

    return topic_name, top_words

# Deploy
st.markdown(
    "<h1 style='text-align: center;'>NLP Topic Model</h1>",
    unsafe_allow_html=True
)
text=st.text_area("Context",placeholder="Write your Context...",height=200)

if st.button("Predict"):
    if text.strip():
        topic_name, top_words = detect_new_text_topic(text)
        st.success(f"Best Topic: {topic_name}")

        st.subheader("Top 10 Words in This Topic:")
        for w in top_words:
            st.write("•", w)
    else:
        st.error("Please enter some text.")