import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline
import time

# Set up the page with improved layout and styling
st.set_page_config(
    page_title="Customer Support AI Assistant", 
    layout="wide",
    page_icon="🤖"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #e8f4f8;
    }
    .report {
        padding: 15px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title("🤖 Customer Support AI Assistant")
st.markdown("""
    This application helps classify customer support queries and generate responses using AI.
    Upload your dataset to train the classifier or use the chatbot for immediate assistance.
    """)

# Sidebar for navigation and info
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Features:**
    - CSV data upload & preprocessing
    - Query classification (SVM model)
    - AI response generation
    - Interactive chatbot
    """)
    
    st.markdown("---")
    st.markdown("""
    **Instructions:**
    1. Upload your customer support dataset (CSV)
    2. View cleaned data and model metrics
    3. Use the chatbot for real-time queries
    """)

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "🔍 Query Classifier", "💬 AI Chatbot"])

with tab1:
    st.header("Data Processing & Model Training")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Customer Support CSV", type="csv", 
                                   help="Should contain 'message' and 'category' columns")
    
    if uploaded_file:
        with st.spinner("Processing data..."):
            df = pd.read_csv(uploaded_file)
            
            # Show raw data with expander
            with st.expander("View Raw Data", expanded=True):
                st.dataframe(df.head())
            
            # Data cleaning
            st.subheader("Data Cleaning")
            
            # Convert text columns to lowercase
            df['channel'] = df['channel'].astype(str).str.lower()
            df['message'] = df['message'].astype(str).str.lower()
            df['category'] = df['category'].astype(str).str.lower()

            # Function to remove HTML tags
            def remove_html_tags(text):
                return re.sub(r'<.*?>', '', text)

            # Function to remove stopwords
            stopwords = ['a','an','the','and','or','is','are','was','were','in','on','at','of','to','for','it','how','can','i']
            def remove_stopwords(text):
                words = re.findall(r'\b\w+\b', text.lower())
                return ' '.join([w for w in words if w not in stopwords])

            # Apply cleaning functions
            df['message'] = df['message'].apply(remove_html_tags)
            df['message'] = df['message'].apply(remove_stopwords)

            with st.expander("View Cleaned Data"):
                st.dataframe(df.head())
            
            # Model training
            st.subheader("Model Training")
            
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(max_features=5000)
            X = vectorizer.fit_transform(df['message'])
            y = df['category']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            # Train an SVM classifier
            model = SVC(kernel='linear', probability=True)
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            
            with st.expander("Model Performance Metrics"):
                st.markdown("**Classification Report**")
                st.code(classification_report(y_test, y_pred))
                
                st.markdown("**Confusion Matrix**")
                st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))
            
            st.success("Model training completed successfully!")

with tab2:
    st.header("Query Classification")
    
    if 'model' not in st.session_state or 'vectorizer' not in st.session_state:
        st.warning("Please upload and process data in the 'Data Analysis' tab first.")
    else:
        query = st.text_area("Enter customer query:", 
                            placeholder="Type your customer support query here...",
                            height=150)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Classify Query"):
                if query.strip():
                    with st.spinner("Analyzing query..."):
                        # Clean query
                        cleaned_query = remove_html_tags(query)
                        cleaned_query = remove_stopwords(cleaned_query)
                        X_query = st.session_state.vectorizer.transform([cleaned_query])

                        # Predict category
                        predicted_category = st.session_state.model.predict(X_query)[0]
                        
                        st.markdown(f"""
                        <div class="report">
                            <h4>Classification Result</h4>
                            <p><strong>Query:</strong> {query[:100]}...</p>
                            <p><strong>Predicted Category:</strong> <span style="color: #4CAF50; font-weight: bold;">{predicted_category}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Please enter a query first.")
        
        with col2:
            if st.button("Generate Response"):
                if query.strip():
                    with st.spinner("Generating response..."):
                        # Load text generation model
                        @st.cache_resource
                        def load_generator():
                            return pipeline("text-generation", model="distilgpt2")
                        generator = load_generator()
                        
                        # Generate response
                        prompt = f"Customer query: {query}. Response:"
                        response = generator(prompt, max_length=100, num_return_sequences=1)
                        
                        st.markdown(f"""
                        <div class="report">
                            <h4>AI Generated Response</h4>
                            <p>{response[0]['generated_text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Please enter a query first.")

with tab3:
    st.header("AI Support Chatbot")
    st.markdown("""
    Chat with our AI assistant for immediate customer support.
    The bot can answer questions based on your uploaded data or general knowledge.
    """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Simulate stream of response with milliseconds delay
            # In a real app, you would replace this with Bard API call
            assistant_response = f"I'm your AI assistant. You asked: '{prompt}'. I would normally consult the Bard API for a detailed response to this query."
            
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Store model and vectorizer in session state for tab access
if uploaded_file and 'model' not in st.session_state:
    st.session_state.model = model
    st.session_state.vectorizer = vectorizer
    
