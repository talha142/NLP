import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline
from wordcloud import WordCloud

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
    .sidebar .section {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================
# SIDEBAR CONTROLS
# ==============================================
with st.sidebar:
    st.markdown('<div class="header"><h2>⚙️ Controls</h2></div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], 
                                   help="File should contain 'message' and 'category' columns")
    
    # EDA settings
    with st.expander("📊 EDA Settings", expanded=True):
        eda_category_limit = st.slider("Top Categories to Show", 5, 20, 10)
        wordcloud_width = st.slider("WordCloud Width", 400, 1000, 800)
        wordcloud_height = st.slider("WordCloud Height", 200, 800, 400)
    
    # Model settings
    with st.expander("🤖 Model Settings"):
        test_size = st.slider("Test Size (%)", 10, 40, 20)
        max_features = st.slider("Max Features", 1000, 10000, 5000, step=500)
    
    # App info
    st.markdown("---")
    st.markdown("""
    **📝 About this app:**
    - Classifies customer support messages
    - Generates suggested responses
    - Provides EDA insights
    """)

# ==============================================
# MAIN CONTENT
# ==============================================
st.markdown('<div class="header"><h1>💬 Customer Support NLP Assistant</h1></div>', unsafe_allow_html=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # ==============================================
    # EDA SECTION
    # ==============================================
    st.markdown("## 🔍 Exploratory Data Analysis")
    
    with st.expander("📊 Basic Statistics", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", len(df))
        with col2:
            st.metric("Unique Categories", df['category'].nunique())
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.dataframe(df.describe(include='object').style.background_gradient(cmap='Blues'))
    
    # Category Distribution
    with st.expander("📈 Category Distribution"):
        fig, ax = plt.subplots()
        top_categories = df['category'].value_counts().nlargest(eda_category_limit)
        sns.barplot(x=top_categories.values, y=top_categories.index, palette="viridis", ax=ax)
        ax.set_title(f"Top {eda_category_limit} Categories")
        st.pyplot(fig)
    
    # Word Cloud
    with st.expander("☁️ Word Cloud"):
        all_text = " ".join(msg for msg in df['message'])
        wordcloud = WordCloud(width=wordcloud_width, height=wordcloud_height, 
                            background_color='white').generate(all_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    
    # ==============================================
    # MODEL SECTION
    # ==============================================
    st.markdown("## 🤖 AI Model")
    
    with st.spinner("Processing data and training model..."):
        # Data cleaning
        df['message'] = df['message'].astype(str).str.lower()
        df['category'] = df['category'].astype(str).str.lower()
        
        def clean_text(text):
            text = re.sub(r'<.*?>', '', text)
            stopwords = ['a','an','the','and','or','is','are','was','were']
            words = re.findall(r'\b\w+\b', text.lower())
            return ' '.join([w for w in words if w not in stopwords])
        
        df['cleaned_message'] = df['message'].apply(clean_text)
        
        # Train model
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(df['cleaned_message'])
        y = df['category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)
        
        # Model evaluation
        y_pred = model.predict(X_test)
        
        st.success("Model trained successfully!")
        
        # Show metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", X_train.shape[0])
            st.metric("Test Samples", X_test.shape[0])
        with col2:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
            st.metric("Features Used", X.shape[1])
    
    # ==============================================
    # PREDICTION INTERFACE
    # ==============================================
    st.markdown("## 💬 Try It Out")
    
    query = st.text_area("Enter a customer message:", height=150)
    
    if st.button("Analyze Message"):
        if query.strip():
            with st.spinner("Analyzing..."):
                # Clean and predict
                cleaned = clean_text(query)
                X_q = vectorizer.transform([cleaned])
                category = model.predict(X_q)[0]
                proba = model.predict_proba(X_q).max()
                
                # Generate response
                generator = pipeline("text-generation", model="distilgpt2")
                response = generator(
                    f"Customer message: {query}\nSupport response:",
                    max_length=100,
                    num_return_sequences=1
                )[0]['generated_text']
                
                # Display results
                st.success(f"**Predicted Category:** {category} (Confidence: {proba:.0%})")
                st.markdown("**Suggested Response:**")
                st.info(response.split("Support response:")[1].strip())
        else:
            st.warning("Please enter a message first")

else:
    st.info("👈 Please upload a CSV file to begin analysis")
    st.image("https://via.placeholder.com/800x400?text=Upload+your+customer+support+data+CSV", 
             use_column_width=True)
