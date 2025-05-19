import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer  # Corrected spelling
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline

# Configure page
st.set_page_config(
    page_title="Customer Support NLP Assistant",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .header {
        color: #2c3e50;
        padding-bottom: 1rem;
        border-bottom: 2px solid #3498db;
    }
    .section {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .dataframe {
        width: 100%;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stTextArea>div>div>textarea {
        min-height: 150px;
    }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="header"><h1>💬 Customer Support NLP Assistant</h1></div>', unsafe_allow_html=True)
st.caption("Automatically classify customer queries and generate responses using AI")

# Sidebar for file upload and info
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Should contain 'message' and 'category' columns")
    
    st.header("About")
    st.info("""
    This app helps customer support teams:
    - Classify incoming messages
    - Generate suggested responses
    - Analyze support trends
    """)

# Main content
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Data Preview Section
        with st.expander("📊 Data Preview", expanded=True):
            st.dataframe(df.head())
        
        # Data Processing
        with st.spinner("Processing data..."):
            # Clean data
            df['message'] = df['message'].astype(str).str.lower()
            df['category'] = df['category'].astype(str).str.lower()
            
            def clean_text(text):
                text = re.sub(r'<.*?>', '', text)  # Remove HTML
                stopwords = ['a','an','the','and','or','is','are','was','were']
                words = re.findall(r'\b\w+\b', text.lower())
                return ' '.join([w for w in words if w not in stopwords])
            
            df['cleaned_message'] = df['message'].apply(clean_text)
            
            # Train model
            vectorizer = TfidfVectorizer(max_features=5000)
            X = vectorizer.fit_transform(df['cleaned_message'])
            y = df['category']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = SVC(kernel='linear', probability=True)
            model.fit(X_train, y_train)
            
            st.success("Data processed and model trained successfully!")

        # Model Evaluation
        with st.expander("📈 Model Performance"):
            y_pred = model.predict(X_test)
            st.text(classification_report(y_test, y_pred))
            st.write("Confusion Matrix:")
            st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

        # Query Section
        st.markdown("---")
        st.subheader("Try It Out")
        
        col1, col2 = st.columns(2)
        
        with col1:
            query = st.text_area("Enter a customer message:", height=150)
            
            if st.button("Analyze Message"):
                if query.strip():
                    with st.spinner("Analyzing..."):
                        # Clean and predict
                        cleaned = clean_text(query)
                        X_q = vectorizer.transform([cleaned])
                        category = model.predict(X_q)[0]
                        proba = model.predict_proba(X_q).max()
                        
                        # Load generator only when needed
                        generator = pipeline("text-generation", model="distilgpt2")
                        prompt = f"Customer message: {query}\nSupport response:"
                        response = generator(prompt, max_length=100, num_return_sequences=1)
                        
                        st.session_state.result = {
                            'category': category,
                            'confidence': f"{proba:.0%}",
                            'response': response[0]['generated_text']
                        }
                else:
                    st.warning("Please enter a message first")
        
        with col2:
            if 'result' in st.session_state:
                st.markdown("### Results")
                st.metric("Predicted Category", st.session_state.result['category'])
                st.metric("Confidence", st.session_state.result['confidence'])
                
                st.markdown("### Suggested Response")
                st.info(st.session_state.result['response'])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()

else:
    st.info("👈 Please upload a CSV file to get started")
