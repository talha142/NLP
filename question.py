# app.py - Main Streamlit App

import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(page_title="Query Classifier App", layout="wide")

# Title and Welcome Message
st.title("💬 Query Classifier and Responder")
st.markdown("""
Welcome to the **Query Classifier App**! This app helps you:
- 📂 Upload and clean your dataset
- 📊 Explore it with EDA (Matplotlib & Seaborn)
- 🤖 Train a machine learning model (SVM)
- 🧠 Predict categories of new queries
- 💡 Generate auto-responses using GPT2

**Built with Streamlit, Scikit-learn, and HuggingFace Transformers** 🚀
""")

st.info("👈 Use the sidebar to navigate through different stages of the application.")

# eda_cleaning.py - Page 2: EDA & Cleaning

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

st.set_page_config(page_title="EDA & Cleaning | Query Classifier", layout="wide")
st.title("📊 Exploratory Data Analysis (EDA) & Cleaning")

stopwords = ['a','an','the','and','or','is','are','was','were','in','on','at','of','to','for','it','how','can','i']

def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)

def remove_stopwords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return ' '.join([w for w in words if w not in stopwords])

def clean_dataframe(df):
    df['message'] = df['message'].astype(str).str.lower()
    df['message'] = df['message'].apply(remove_html_tags)
    df['message'] = df['message'].apply(remove_stopwords)
    return df

uploaded_file = st.file_uploader("Upload dataset for EDA (CSV with 'message' and 'category')", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📌 Raw Data Preview")
    st.dataframe(df.head())

    st.subheader("🔍 Basic Statistics")
    st.write(df.describe(include='all'))

    st.subheader("🧹 Cleaning Data")
    df_cleaned = clean_dataframe(df)
    st.write("✅ Data cleaned successfully.")

    st.subheader("📈 Category Distribution")
    plt.figure(figsize=(10, 5))
    sns.countplot(y='category', data=df_cleaned, order=df_cleaned['category'].value_counts().index)
    plt.title("Count of Categories")
    st.pyplot(plt)

    st.session_state['cleaned_df'] = df_cleaned
else:
    st.info("👈 Upload a CSV file to begin analysis.")

# model_training.py - Page 3: Model Training (already created)
# (kept as-is)

# query_classification.py - Page 4: Query Classification

import streamlit as st
import re

st.set_page_config(page_title="Query Classification", layout="wide")
st.title("🧠 Predict Query Category")

def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)

def remove_stopwords(text):
    stopwords = ['a','an','the','and','or','is','are','was','were','in','on','at','of','to','for','it','how','can','i']
    words = re.findall(r'\b\w+\b', text.lower())
    return ' '.join([w for w in words if w not in stopwords])

model = st.session_state.get('model', None)
vectorizer = st.session_state.get('vectorizer', None)

if model and vectorizer:
    user_input = st.text_input("Enter your query:")
    if user_input:
        cleaned = remove_stopwords(remove_html_tags(user_input))
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        st.success(f"🔮 Predicted Category: **{prediction}**")
else:
    st.warning("⚠️ Please train a model first in the Model Training tab.")

# generate_response.py - Page 5: Response Generation

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Response Generator", layout="wide")
st.title("💡 Auto-Generate Response")

generator = pipeline("text-generation", model="distilgpt2")

user_query = st.text_area("Enter user query to generate response:")
if user_query:
    prompt = f"Customer query: {user_query}. Response:"
    response = generator(prompt, max_length=100, num_return_sequences=1)
    st.subheader("🤖 Generated Response")
    st.write(response[0]['generated_text'])
