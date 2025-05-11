## Project: Customer Support NLP App (Streamlit multipage)
# File structure:
# - app.py
# - pages/
#     - 1_Data_Preprocessing.py
#     - 2_Model_Training.py
#     - 3_Query_and_Generation.py

# ===== app.py =====
import streamlit as st

st.set_page_config(page_title="Customer Support NLP App", layout="wide")

st.title("Customer Support NLP Application")
st.markdown("""
Welcome to the multi-page NLP app for customer support analysis.

**Navigation:**
- Use the sidebar to go through each step:
  1. Data Preprocessing
  2. Model Training & Evaluation
  3. Query Classification & Text Generation
""")

# pages are automatically discovered by Streamlit Multipage

# ===== pages/1_Data_Preprocessing.py =====
import streamlit as st
import pandas as pd
import re

st.title("Step 1: Data Cleaning and Preprocessing")

uploaded_file = st.file_uploader("Upload your customer support CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.write(data.head())

    # lowercase specified columns
    for col in ['channel', 'message', 'category']:
        if col in data.columns:
            data[col] = data[col].str.lower()

    # remove HTML tags
    def remove_html_tags(text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    data['message'] = data['message'].apply(remove_html_tags)

    # define stopwords set
    stopwords = {'a','an','the','and','or','is','are','was','were','in','on','at','of','to','for','it','how','can','i'}
    def remove_stopwords(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return ' '.join([w for w in words if w not in stopwords])

    data['message'] = data['message'].apply(remove_stopwords)
    
    st.subheader("Cleaned Data Preview")
    st.write(data.head())

    # store cleaned data in session state
    st.session_state.cleaned_data = data
else:
    st.info("Please upload a CSV file.")

# ===== pages/2_Model_Training.py =====
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

st.title("Step 2: Model Training and Evaluation")

if 'cleaned_data' in st.session_state:
    data = st.session_state.cleaned_data
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['message'])
    y = data['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    # cache vectorizer and model
    st.session_state.vectorizer = vectorizer
    st.session_state.model = model
    st.success("Model trained and evaluated successfully.")
else:
    st.warning("Please complete data preprocessing first.")

# ===== pages/3_Query_and_Generation.py =====
import streamlit as st
import re
from transformers import pipeline

st.title("Step 3: Query Classification & Text Generation")

if 'model' in st.session_state and 'vectorizer' in st.session_state:
    model = st.session_state.model
    vectorizer = st.session_state.vectorizer

    @st.cache_resource
    def load_generator():
        return pipeline("text-generation", model="distilgpt2")
    generator = load_generator()

    query = st.text_area("Enter a customer query:")
    if st.button("Predict & Generate Response"):
        if query.strip():
            # preprocess query
            def remove_html_tags(text):
                return re.sub(re.compile('<.*?>'), '', text)
            stopwords = {'a','an','the','and','or','is','are','was','were','in','on','at','of','to','for','it','how','can','i'}
            def remove_stopwords(text):
                words = re.findall(r'\b\w+\b', text.lower())
                return ' '.join([w for w in words if w not in stopwords])

            cleaned = remove_stopwords(remove_html_tags(query))
            vec = vectorizer.transform([cleaned])
            pred_cat = model.predict(vec)[0]

            st.subheader("Predicted Category")
            st.info(pred_cat)

            prompt = f"Category: {pred_cat}\nCustomer Query: {query}\nResponse:"
            resp = generator(prompt, max_length=100, num_return_sequences=1)

            st.subheader("Generated Response")
            st.success(resp[0]['generated_text'])
        else:
            st.warning("Please enter a query.")
else:
    st.warning("Please complete model training first.")
