import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

df = load_data()

# -----------------------------
# Preprocessing
# -----------------------------
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

X_dense = X.toarray()

# -----------------------------
# Train Lasso Model
# -----------------------------
model = Lasso(alpha=0.1)
model.fit(X_dense, y)

# Feature info
total_features = X_dense.shape[1]
non_zero = np.sum(model.coef_ != 0)
reduction = ((total_features - non_zero) / total_features) * 100

# -----------------------------
# UI Design
# -----------------------------
st.title("📩 SMS Spam ML App")
st.write("Detect whether a message is **Spam or Ham** using ML + Lasso")

st.sidebar.header("📊 Model Info")
st.sidebar.write(f"Total Features: {total_features}")
st.sidebar.write(f"Selected Features: {non_zero}")
st.sidebar.write(f"Reduction: {reduction:.2f}%")

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area("✉️ Enter your SMS message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        vec = vectorizer.transform([user_input]).toarray()
        pred = model.predict(vec)

        if pred[0] > 0.5:
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is HAM (Not Spam)")
