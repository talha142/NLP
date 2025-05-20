import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
from openai import OpenAI
import os

# ------------------------------
# App Configuration
# ------------------------------
st.set_page_config(
    page_title="AI Query Assistant",
    page_icon="🤖",
    layout="wide"
)

# ------------------------------
# Session State Initialization
# ------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None

# ------------------------------
# Utility Functions
# ------------------------------
def clean_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def initialize_openai_client(api_key):
    """Initialize OpenAI client with error handling"""
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {str(e)}")
        return None

# ------------------------------
# Machine Learning Functions
# ------------------------------
def train_model(df):
    """Train classification model"""
    try:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df['message'])
        y = df['category']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        return model, vectorizer, report, cm
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None, None, None

def predict_category(text, model, vectorizer):
    """Predict category for new text"""
    try:
        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        return pred, proba
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

# ------------------------------
# OpenAI Functions
# ------------------------------
def generate_ai_response(query, category, client):
    """Generate response using OpenAI API"""
    if not client:
        return "OpenAI client not initialized"
    
    prompt = f"""
    As a customer service assistant, provide a helpful and professional response 
    to this {category} category query. Keep the response concise (1-2 paragraphs).
    
    Query: {query}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful customer service assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ------------------------------
# UI Components
# ------------------------------
def show_data_upload():
    """Upload data UI"""
    st.header("📤 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with 'message' and 'category' columns",
        type="csv"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['message'] = df['message'].apply(clean_text)
        st.session_state.df = df
        st.success(f"Data loaded successfully! {len(df)} records found.")
        
        with st.expander("View Data Preview"):
            st.dataframe(df.head())

def show_data_exploration():
    """Data exploration UI"""
    st.header("🔍 Explore Your Data")
    
    if st.session_state.df is None:
        st.warning("Please upload data first")
        return
    
    df = st.session_state.df
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Queries", len(df))
    col2.metric("Unique Categories", df['category'].nunique())
    col3.metric("Avg Query Length", round(df['message'].str.len().mean(), 1))
    
    # Category distribution
    st.subheader("Category Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    df['category'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Word cloud
    st.subheader("Common Words")
    text = " ".join(df['message'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def show_model_training():
    """Model training UI"""
    st.header("⚙️ Train Classification Model")
    
    if st.session_state.df is None:
        st.warning("Please upload data first")
        return
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model, vectorizer, report, cm = train_model(st.session_state.df)
            
            if model:
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                st.success("Model trained successfully!")
                
                # Show metrics
                st.subheader("Model Performance")
                st.write(f"Accuracy: {report['accuracy']:.2%}")
                
                # Classification report
                st.subheader("Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0))
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=model.classes_, yticklabels=model.classes_)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig)

def show_query_analysis():
    """Query analysis and response UI"""
    st.header("💬 Analyze & Respond")
    
    if st.session_state.model is None:
        st.warning("Please train the model first")
        return
    
    query = st.text_area("Enter customer query:", height=150)
    
    if st.button("Analyze & Generate Response"):
        if not query:
            st.error("Please enter a query")
            return
        
        with st.spinner("Processing..."):
            # Classify query
            pred, proba = predict_category(query, st.session_state.model, st.session_state.vectorizer)
            
            if pred is None:
                return
            
            st.success(f"**Predicted Category:** {pred}")
            
            # Show probabilities
            st.subheader("Category Confidence")
            prob_df = pd.DataFrame({
                'Category': st.session_state.model.classes_,
                'Probability': proba
            }).sort_values('Probability', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Probability', y='Category', data=prob_df.head(5), palette='viridis', ax=ax)
            plt.xlim(0, 1)
            st.pyplot(fig)
            
            # Generate AI response
            if st.session_state.openai_client:
                response = generate_ai_response(query, pred, st.session_state.openai_client)
                st.subheader("Suggested Response")
                st.markdown(f"""
                <div style="
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid #1e90ff;
                    margin-top: 10px;
                ">
                {response}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("OpenAI client not initialized - cannot generate response")

# ------------------------------
# Sidebar Configuration
# ------------------------------
def configure_sidebar():
    """Sidebar configuration"""
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to",
            ["📤 Upload Data", "🔍 Explore Data", "⚙️ Train Model", "💬 Analyze & Respond"]
        )
        
        st.markdown("---")
        st.subheader("OpenAI Configuration")
        api_key = st.text_input(
            "Enter OpenAI API Key",
            type="password",
            help="Get your key from platform.openai.com"
        )
        
        if api_key:
            st.session_state.openai_client = initialize_openai_client(api_key)
            if st.session_state.openai_client:
                st.success("OpenAI client initialized!")
        
        st.markdown("---")
        st.caption("AI Query Assistant v1.0")

# ------------------------------
# Main App
# ------------------------------
def main():
    """Main app function"""
    st.title("🤖 AI Query Assistant")
    st.markdown("""
    Classify customer queries and generate AI-powered responses.
    """)
    
    configure_sidebar()
    
    # Get current page from sidebar
    page = st.sidebar.radio("", ["📤 Upload Data", "🔍 Explore Data", "⚙️ Train Model", "💬 Analyze & Respond"], label_visibility="collapsed")
    
    # Page routing
    if page == "📤 Upload Data":
        show_data_upload()
    elif page == "🔍 Explore Data":
        show_data_exploration()
    elif page == "⚙️ Train Model":
        show_model_training()
    elif page == "💬 Analyze & Respond":
        show_query_analysis()

if __name__ == "__main__":
    main()
