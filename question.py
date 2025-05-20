# welcome.py

import streamlit as st

# Set Streamlit page configuration
st.set_page_config(page_title="Welcome | Query Classifier", layout="wide")

# Title and introduction
st.title("🧠 Smart Query Classifier & Responder")

# Welcome Message
st.markdown("""
### 👋 Welcome to the Smart Query Classifier & Responder App

This application helps you to:

- 📂 Upload a dataset containing customer messages.
- 📊 Explore the data through visualizations.
- 🧠 Train a machine learning model (SVM) to classify message categories.
- 🔍 Predict the category of new queries.
- 🤖 Generate automated responses using a transformer-based model.

### 🚀 How to Use:
1. Navigate to **Upload File** to provide your dataset.
2. Go to **EDA** to visually analyze your data.
3. Proceed to **Model Training** to build your text classification model.
4. Try **Query Classification** to classify a new message.
5. Use **Generate Response** to get a smart AI-based reply.

### 🛠 Built With:
- Streamlit
- Scikit-learn
- Hugging Face Transformers
- Pandas & Plotly

Start by selecting an option from the sidebar!
""")
