import streamlit as st
import pandas as pd
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download resources
nltk.download('stopwords')
nltk.download('wordnet')

# load dataset
df = pd.read_csv("fake_news_dataset.csv")
df.columns = ["Text", "Label"]

# preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

df['clean_text'] = df['Text'].apply(preprocess)

# TF-IDF + Naive Bayes
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(df['clean_text'])

model = MultinomialNB()
model.fit(X_vec, df['Label'])

# ---------------- UI ----------------
st.title("📰 Fake News Detection App")

st.write("Enter a news sentence and check whether it is Real or Fake")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    clean = preprocess(user_input)
    vec = vectorizer.transform([clean])
    result = model.predict(vec)[0]

    st.subheader("Prediction:")
    if result == "fake":
        st.error("🚫 This is FAKE news")
    else:
        st.success("✅ This is REAL news")
