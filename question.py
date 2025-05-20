import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline

# ------------------------------
# Utility functions for cleaning
# ------------------------------

def remove_html_tags(text):
    """Remove HTML tags using regex"""
    return re.sub(r'<.*?>', '', text)

stopwords = set(['a','an','the','and','or','is','are','was','were','in','on','at','of','to','for','it','how','can','i'])

def remove_stopwords(text):
    """Remove common stopwords from text"""
    words = re.findall(r'\b\w+\b', text.lower())
    filtered = [w for w in words if w not in stopwords]
    return ' '.join(filtered)

def clean_dataframe(df):
    """Clean dataframe columns and text"""
    if 'channel' in df.columns:
        df['channel'] = df['channel'].astype(str).str.lower()
    df['message'] = df['message'].astype(str).str.lower()
    df['category'] = df['category'].astype(str).str.lower()
    df['message'] = df['message'].apply(remove_html_tags)
    df['message'] = df['message'].apply(remove_stopwords)
    return df

# ------------------------------
# Model training and prediction
# ------------------------------

def train_model(df):
    """Train an SVM classifier on cleaned data"""
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['message'])
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    matrix = confusion_matrix(y_test, y_pred)

    return model, vectorizer, report, matrix

def predict_category(query, model, vectorizer):
    """Predict category from a user query"""
    cleaned = remove_stopwords(remove_html_tags(query))
    X_query = vectorizer.transform([cleaned])
    prediction = model.predict(X_query)[0]
    return prediction

# ------------------------------
# Text generation with transformers
# ------------------------------

@st.cache_resource(show_spinner=False)
def load_generator():
    """Load GPT2 text generation pipeline"""
    return pipeline("text-generation", model="distilgpt2")

def generate_response(query, generator):
    """Generate response text for a query"""
    prompt = f"Customer query: {query}. Response:"
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

# ------------------------------
# Streamlit app main
# ------------------------------

# Set page config once at the top
st.set_page_config(page_title="Query Classification & Response App", layout="wide")

st.title("💬 Query Classification and Auto Response Application")

st.markdown("""
This app allows you to:
- Upload and clean chat/query data
- Explore and visualize categories
- Train an SVM classifier on the data
- Classify new queries
- Generate automatic responses using GPT-2
Use the sidebar to navigate through the steps.
""")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a step:", 
    ["Upload & Clean Data", "EDA & Visualization", "Train Model", "Classify Query", "Generate Response"])

# Session state initialization
if 'df_cleaned' not in st.session_state:
    st.session_state['df_cleaned'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'vectorizer' not in st.session_state:
    st.session_state['vectorizer'] = None
if 'generator' not in st.session_state:
    st.session_state['generator'] = load_generator()

# ------------------------------
# Page 1: Upload & Clean Data
# ------------------------------

if page == "Upload & Clean Data":
    st.header("📂 Upload Your Dataset")

    uploaded_file = st.file_uploader("Upload CSV with columns: 'message', 'category' (and optionally 'channel')", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Raw Data Sample")
        st.dataframe(df.head())

        if st.button("Clean Data"):
            try:
                df_cleaned = clean_dataframe(df)
                st.session_state['df_cleaned'] = df_cleaned
                st.success("Data cleaned successfully!")
                st.subheader("Cleaned Data Sample")
                st.dataframe(df_cleaned.head())
            except Exception as e:
                st.error(f"Error during cleaning: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")

# ------------------------------
# Page 2: EDA & Visualization
# ------------------------------

elif page == "EDA & Visualization":
    st.header("📊 Exploratory Data Analysis")

    df = st.session_state['df_cleaned']
    if df is None:
        st.warning("Please upload and clean data first.")
    else:
        st.subheader("Data Info")
        st.write(df.info())

        st.subheader("Category Distribution")
        plt.figure(figsize=(10, 6))
        sns.countplot(y='category', data=df, order=df['category'].value_counts().index)
        plt.title("Count of Messages by Category")
        plt.xlabel("Count")
        plt.ylabel("Category")
        st.pyplot(plt)

        st.subheader("Sample Messages")
        selected_cat = st.selectbox("Select category to view messages", options=df['category'].unique())
        sample_texts = df[df['category'] == selected_cat]['message'].sample(min(5, len(df))).values
        for i, text in enumerate(sample_texts, 1):
            st.write(f"{i}. {text}")

# ------------------------------
# Page 3: Train Model
# ------------------------------

elif page == "Train Model":
    st.header("🤖 Train Classification Model")

    df = st.session_state['df_cleaned']
    if df is None:
        st.warning("Please upload and clean data first.")
    else:
        if st.button("Train SVM Model"):
            with st.spinner("Training..."):
                model, vectorizer, report, matrix = train_model(df)
                st.session_state['model'] = model
                st.session_state['vectorizer'] = vectorizer

                st.success("Model trained successfully!")

                st.subheader("Classification Report")
                st.text(report)

                st.subheader("Confusion Matrix")
                plt.figure(figsize=(8,6))
                sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
                st.pyplot(plt)
        else:
            st.info("Click the button to train the model.")

# ------------------------------
# Page 4: Classify Query
# ------------------------------

elif page == "Classify Query":
    st.header("🧠 Query Category Prediction")

    model = st.session_state.get('model')
    vectorizer = st.session_state.get('vectorizer')

    if model is None or vectorizer is None:
        st.warning("Please train the model first.")
    else:
        query = st.text_input("Enter your query text to classify:")
        if query:
            prediction = predict_category(query, model, vectorizer)
            st.success(f"Predicted Category: **{prediction}**")

# ------------------------------
# Page 5: Generate Response
# ------------------------------

elif page == "Generate Response":
    st.header("💡 Auto Response Generation")

    generator = st.session_state['generator']
    query = st.text_area("Enter query text for response generation:")

    if query:
        with st.spinner("Generating response..."):
            response = generate_response(query, generator)
            st.subheader("Generated Response")
            st.write(response)
