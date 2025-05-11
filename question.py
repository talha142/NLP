import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline

# Set up the app
st.set_page_config(page_title="Enhanced Customer Support NLP App", layout="wide")
st.title("Enhanced Customer Support NLP Application")

# Initialize session state variables
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None

# Navigation
page = st.sidebar.selectbox("Select Step", 
                           ["Home", 
                            "1. Data Preprocessing", 
                            "2. Model Training & Evaluation", 
                            "3. Query Processing & Response Generation"])

if page == "Home":
    st.markdown("""
    Welcome to the enhanced Customer Support NLP application.
    
    **Key Improvements:**
    - Better category prediction
    - Category-specific response generation
    - Improved preprocessing pipeline
    - More accurate response generation
    
    **Navigation:**
    - Use the sidebar to go through each step:
      1. Data Preprocessing: Clean and prepare your data
      2. Model Training: Train and evaluate the category classifier
      3. Query Processing: Get category predictions and generated responses
    """)

elif page == "1. Data Preprocessing":
    st.title("Step 1: Data Cleaning and Preprocessing")
    
    uploaded_file = st.file_uploader("Upload your customer support CSV file", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Raw Data Preview")
        st.write(data.head())
        
        # Check required columns
        required_cols = ['message', 'category']
        if not all(col in data.columns for col in required_cols):
            st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
        else:
            # Basic cleaning
            data = data.dropna(subset=['message', 'category'])
            
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
            
            st.subheader("Cleaned Data Preview")
            st.write(data.head())
            
            # Show category distribution
            st.subheader("Category Distribution")
            st.bar_chart(data['category'].value_counts())
            
            st.session_state.cleaned_data = data
            st.success("Data preprocessing completed!")
    else:
        st.info("Please upload a CSV file with 'message' and 'category' columns.")

elif page == "2. Model Training & Evaluation":
    st.title("Step 2: Model Training and Evaluation")
    
    if 'cleaned_data' in st.session_state:
        data = st.session_state.cleaned_data
        
        # Vectorization
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(data['message'])
        y = data['category']
        
        st.session_state.vectorizer = vectorizer
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Model training
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)
        
        st.session_state.model = model
        
        # Evaluation
        y_pred = model.predict(X_test)
        
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
        
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))
        
        st.success("Model trained and evaluated successfully!")
    else:
        st.warning("Please complete data preprocessing first.")

elif page == "3. Query Processing & Response Generation":
    st.title("Step 3: Query Processing & Response Generation")
    
    @st.cache_resource
    def load_generator():
        return pipeline("text-generation", model="gpt2")
    
    generator = load_generator()
    
    query = st.text_area("Enter a customer query:", 
                        placeholder="e.g., My order hasn't arrived yet")
    
    if st.button("Process Query"):
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
                probabilities = model.predict_proba(query_vec)[0]
                
                st.subheader("Prediction Results")
                st.write(f"Predicted Category: **{category}**")
                
                # Show top categories
                classes = model.classes_
                top_n = 5
                top_indices = probabilities.argsort()[-top_n:][::-1]
                
                st.write("Top Category Probabilities:")
                for i in top_indices:
                    st.write(f"- {classes[i]}: {probabilities[i]:.2f}")
                
                # Generate response
                st.subheader("Generated Response")
                
                # Category-specific prompts
                prompt_templates = {
                    'shipping': f"Customer question about shipping: {query}\nAssistant response:",
                    'payment': f"Payment-related question: {query}\nCustomer support reply:",
                    'refund': f"Refund request: {query}\nProfessional response:",
                    'product': f"Question about product: {query}\nHelpful answer:",
                    'account': f"Account-related issue: {query}\nSupport response:"
                }
                
                # Use specific prompt if available, otherwise generic
                prompt = prompt_templates.get(category, 
                    f"Customer support question: {query}\nCategory: {category}\nProfessional response:")
                
                response = generator(
                    prompt,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                
                # Clean up the response
                generated_text = response[0]['generated_text']
                # Remove the prompt from the response
                generated_response = generated_text.replace(prompt, "").strip()
                # Remove any incomplete sentences at the end
                generated_response = re.sub(r'[^.!?]*$', '', generated_response).strip()
                
                st.success(generated_response)
                
                # Show debug info
                with st.expander("Debug Information"):
                    st.write("Full generated text:")
                    st.text(generated_text)
                    st.write("Prompt used:")
                    st.text(prompt)
            else:
                st.warning("Please train the model first in Step 2.")
        else:
            st.warning("Please enter a customer query.")
