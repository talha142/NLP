import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline

# Set up the app
st.set_page_config(page_title="Customer Support NLP App", layout="wide")

# Initialize session state variables
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None

# Home Page
def home_page():
    st.title("Customer Support NLP Application")
    st.markdown("""
    Welcome to the Customer Support NLP application. This tool helps you:
    - Analyze customer support tickets
    - Automatically categorize queries
    - Generate suggested responses
    
    **How to use:**
    1. Upload your customer support data (CSV file)
    2. Preprocess and clean the data
    3. Train the classification model
    4. Generate responses to new queries
    """)
    
    # File uploader on home page
    uploaded_file = st.file_uploader("Upload your customer support CSV file", type="csv", key="home_uploader")
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = data
            st.success("File uploaded successfully!")
            
            # Show quick preview
            with st.expander("Show data preview"):
                st.write("First 5 rows:")
                st.dataframe(data.head())
                
                # Check for required columns
                required_cols = ['message', 'category']
                if all(col in data.columns for col in required_cols):
                    st.write("✅ Required columns ('message', 'category') found")
                else:
                    st.error(f"Missing required columns. Need: {', '.join(required_cols)}")
                
                # Show basic stats
                st.write("\nBasic statistics:")
                st.write(f"- Total records: {len(data)}")
                st.write(f"- Categories: {data['category'].nunique()}")
                st.write(f"- Sample categories: {', '.join(data['category'].unique()[:5])}")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Data Preprocessing Page
def data_preprocessing():
    st.title("Step 1: Data Cleaning and Preprocessing")
    
    if 'uploaded_data' in st.session_state:
        data = st.session_state.uploaded_data.copy()
        
        # Check required columns
        required_cols = ['message', 'category']
        if not all(col in data.columns for col in required_cols):
            st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
            return
        
        st.subheader("Raw Data Preview")
        st.write(data.head())
        
        # Basic cleaning
        with st.spinner("Cleaning data..."):
            # Convert to lowercase
            for col in ['message', 'category']:
                data[col] = data[col].astype(str).str.lower()
            
            # Remove HTML tags
            def remove_html_tags(text):
                clean = re.compile('<.*?>')
                return re.sub(clean, '', text)
            
            data['message'] = data['message'].apply(remove_html_tags)
            
            # Remove special characters and numbers
            def clean_text(text):
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                return re.sub(r'\s+', ' ', text).strip()
            
            data['message'] = data['message'].apply(clean_text)
            
            # Custom stopwords removal
            stopwords = {'a', 'an', 'the', 'and', 'or', 'is', 'are', 'was', 'were', 
                        'in', 'on', 'at', 'of', 'to', 'for', 'it', 'how', 'can', 'i'}
            
            def remove_stopwords(text):
                words = re.findall(r'\b\w+\b', text.lower())
                return ' '.join([word for word in words if word not in stopwords])
            
            data['message'] = data['message'].apply(remove_stopwords)
            
            # Drop rows with empty messages
            data = data[data['message'].str.strip() != '']
            
            st.session_state.cleaned_data = data
        
        st.subheader("Cleaned Data Preview")
        st.write(data.head())
        
        # Show category distribution
        st.subheader("Category Distribution")
        st.bar_chart(data['category'].value_counts())
        
        st.success("Data preprocessing completed!")
    else:
        st.info("Please upload a CSV file on the Home page first.")

# Model Training Page
def model_training():
    st.title("Step 2: Model Training and Evaluation")
    
    if 'cleaned_data' in st.session_state:
        data = st.session_state.cleaned_data
        
        # Vectorization
        with st.spinner("Vectorizing text data..."):
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X = vectorizer.fit_transform(data['message'])
            y = data['category']
            
            st.session_state.vectorizer = vectorizer
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Model training
        with st.spinner("Training SVM model..."):
            model = SVC(kernel='linear', probability=True)
            model.fit(X_train, y_train)
            
            st.session_state.model = model
        
        # Evaluation
        st.subheader("Model Evaluation")
        
        y_pred = model.predict(X_test)
        
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        
        st.success("Model trained and evaluated successfully!")
    else:
        st.warning("Please complete data preprocessing first.")

# Text Generation Page
def text_generation():
    st.title("Step 3: Text Generation")
    
    @st.cache_resource
    def load_generator():
        return pipeline("text-generation", model="distilgpt2")
    
    generator = load_generator()
    
    query = st.text_area("Enter a customer query:", 
                        placeholder="e.g., My order hasn't arrived yet")
    
    if st.button("Generate Response"):
        if query.strip():
            if 'model' in st.session_state and 'vectorizer' in st.session_state:
                # Preprocess the query
                cleaned_query = re.sub(r'[^a-zA-Z\s]', '', query)
                cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip().lower()
                
                # Vectorize and predict
                vectorizer = st.session_state.vectorizer
                model = st.session_state.model
                
                query_vec = vectorizer.transform([cleaned_query])
                category = model.predict(query_vec)[0]
                
                st.subheader("Prediction Results")
                st.write(f"Predicted Category: **{category}**")
                
                # Generate response
                st.subheader("Generated Response")
                
                prompt = f"Customer support question about {category}: {query}\nProfessional response:"
                
                response = generator(
                    prompt,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                
                # Clean up the response
                generated_text = response[0]['generated_text']
                generated_response = generated_text.replace(prompt, "").strip()
                generated_response = re.sub(r'[^.!?]*$', '', generated_response).strip()
                
                st.success(generated_response)
            else:
                st.warning("Please train the model first in Step 2.")
        else:
            st.warning("Please enter a customer query.")

# Main app navigation
pages = {
    "Home": home_page,
    "1. Data Preprocessing": data_preprocessing,
    "2. Model Training": model_training,
    "3. Text Generation": text_generation
}

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
