import streamlit as st

# App configuration
st.set_page_config(page_title="Customer Support NLP App", layout="wide")

# Home Page
st.title("🤖 Customer Support NLP Application")
st.markdown("""
Welcome to the **Customer Support NLP App** – a multi-page Streamlit application designed to help analyze, classify, and generate responses for customer queries using Natural Language Processing (NLP).

---

### 📋 Use this app to:
1. **Explore and Understand Data (EDA)**
2. **Train a Classifier to Predict Review Categories**
3. **Get Help via AI-Powered Response Generation**

👉 Use the sidebar to navigate through each section.
""")

import streamlit as st
import pandas as pd
import re

st.title("📊 Step 1: Data EDA (Exploratory Data Analysis)")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your customer support CSV file", type="csv")

if uploaded_file:
    # Load and preview data
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data Preview")
    st.write(data.head())

    # Lowercasing text columns
    for col in ['channel', 'message', 'category']:
        if col in data.columns:
            data[col] = data[col].astype(str).str.lower()

    # Remove HTML tags from message
    def remove_html_tags(text):
        return re.sub(r'<.*?>', '', text)

    data['message'] = data['message'].apply(remove_html_tags)

    # Remove basic stopwords
    stopwords = {'a','an','the','and','or','is','are','was','were','in','on','at','of','to','for','it','how','can','i'}
    def remove_stopwords(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return ' '.join([w for w in words if w not in stopwords])

    data['message'] = data['message'].apply(remove_stopwords)

    st.subheader("🧼 Cleaned Data Preview")
    st.write(data.head())

    # Save cleaned data to session state
    st.session_state.cleaned_data = data
else:
    st.info("📂 Please upload a CSV file to proceed.")

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

st.title("🧠 Step 2: Review Category Prediction")

# Check if cleaned data is available
if 'cleaned_data' in st.session_state:
    data = st.session_state.cleaned_data

    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['message'])
    y = data['category']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Support Vector Classifier
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Show results
    st.subheader("📈 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("📉 Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    # Store model and vectorizer for use in generation page
    st.session_state.model = model
    st.session_state.vectorizer = vectorizer

    st.success("✅ Model trained successfully!")
else:
    st.warning("⚠️ Please complete Step 1 (EDA) before training the model.")

import streamlit as st
import re
from transformers import pipeline

st.title("🆘 Step 3: AI Help & Response Generator")

# Check for trained model and vectorizer
if 'model' in st.session_state and 'vectorizer' in st.session_state:
    model = st.session_state.model
    vectorizer = st.session_state.vectorizer

    # Load text generator only once (cached)
    @st.cache_resource
    def load_generator():
        return pipeline("text-generation", model="distilgpt2")

    generator = load_generator()

    # User input for query
    query = st.text_area("💬 Enter a customer query for help:", height=150)

    if st.button("🧠 Predict Category & Generate AI Response"):
        if query.strip():
            # Clean the input query
            def remove_html_tags(text):
                return re.sub(r'<.*?>', '', text)

            stopwords = {'a','an','the','and','or','is','are','was','were','in','on','at','of','to','for','it','how','can','i'}
            def remove_stopwords(text):
                words = re.findall(r'\b\w+\b', text.lower())
                return ' '.join([w for w in words if w not in stopwords])

            cleaned_query = remove_stopwords(remove_html_tags(query))

            # Transform and predict
            vec_query = vectorizer.transform([cleaned_query])
            predicted_category = model.predict(vec_query)[0]

            st.subheader("📌 Predicted Category")
            st.info(predicted_category)

            # Prompt for generator
            prompt = f"Category: {predicted_category}\nCustomer Query: {query}\nResponse:"
            response = generator(prompt, max_length=100, num_return_sequences=1)

            st.subheader("💡 AI Generated Response")
            st.success(response[0]['generated_text'])
        else:
            st.warning("❗Please enter a query to get a response.")
else:
    st.warning("⚠️ Model not found. Please train the model first (Step 2).")
