import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import pipeline
from wordcloud import WordCloud
from PIL import Image

# Configure page
st.set_page_config(
    page_title="Customer Support Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
# SIDEBAR CONTROLS
# ==============================================
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Support+AI", width=150)
    st.title("Settings")
    
    uploaded_file = st.file_uploader("📁 Upload Customer Data", type=["csv", "xlsx"])
    
    with st.expander("🧹 Data Cleaning Options"):
        remove_html = st.checkbox("Remove HTML tags", True)
        remove_stopwords = st.checkbox("Remove stop words", True)
        clean_emails = st.checkbox("Remove emails", True)
        clean_urls = st.checkbox("Remove URLs", True)
    
    with st.expander("📈 EDA Settings"):
        heatmap_corr = st.checkbox("Show correlation heatmap", True)
        top_categories = st.slider("Number of top categories", 5, 20, 10)
    
    st.markdown("---")
    st.info("""
    **App Features:**
    - Automated data cleaning
    - Interactive EDA
    - AI-powered classification
    - Response generation
    """)

# ==============================================
# MAIN CONTENT
# ==============================================
st.title("📊 Customer Support Analysis Dashboard")

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # ==============================================
    # DATA CLEANING SECTION
    # ==============================================
    st.header("🧹 Data Cleaning")
    
    with st.expander("Before Cleaning"):
        st.dataframe(df.head())
        st.write("Data shape:", df.shape)
    
    # Cleaning functions
    def clean_text(text):
        text = str(text).lower()
        if remove_html:
            text = re.sub(r'<.*?>', '', text)
        if clean_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        if clean_emails:
            text = re.sub(r'\S+@\S+', '', text)
        if remove_stopwords:
            stopwords = ['a','an','the','and','or','is','are','was','were','in','on','at']
            text = ' '.join([word for word in text.split() if word not in stopwords])
        return text.strip()
    
    with st.spinner("Cleaning data..."):
        df_clean = df.copy()
        df_clean['message'] = df_clean['message'].apply(clean_text)
        df_clean['category'] = df_clean['category'].str.lower().str.strip()
        df_clean = df_clean.dropna(subset=['message', 'category'])
    
    with st.expander("After Cleaning"):
        st.dataframe(df_clean.head())
        st.write("Cleaned data shape:", df_clean.shape)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows removed", f"{len(df) - len(df_clean)}")
        with col2:
            st.metric("Empty values", df_clean.isnull().sum().sum())
    
    # ==============================================
    # EDA SECTION WITH HEATMAP
    # ==============================================
    st.header("🔍 Exploratory Data Analysis")
    
    # Category Distribution
    st.subheader("Category Distribution")
    top_cats = df_clean['category'].value_counts().nlargest(top_categories)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_cats.values, y=top_cats.index, palette="viridis", ax=ax1)
    ax1.set_title(f"Top {top_categories} Categories")
    st.pyplot(fig1)
    
    # Correlation Heatmap (for numerical data)
    if heatmap_corr:
        st.subheader("Feature Correlation")
        
        # Create numerical features for demo
        df_clean['message_length'] = df_clean['message'].apply(len)
        df_clean['word_count'] = df_clean['message'].apply(lambda x: len(x.split()))
        
        # Create correlation matrix
        corr_matrix = df_clean.select_dtypes(include=np.number).corr()
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax2)
        ax2.set_title("Feature Correlation Heatmap")
        st.pyplot(fig2)
    
    # Word Cloud
    st.subheader("Word Cloud")
    all_text = " ".join(msg for msg in df_clean['message'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis("off")
    st.pyplot(fig3)

else:
    st.info("👋 Please upload a customer support dataset to begin analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://via.placeholder.com/400x300?text=Upload+CSV+or+Excel", width=400)
    with col2:
        st.markdown("""
        ### Expected Data Format:
        - `message`: Customer support text
        - `category`: Problem category
        
        **Sample Data:**
        ```csv
        message,category
        "My order hasn't arrived",Shipping
        "The app keeps crashing",Technical
        "How do I reset my password?",Account
        ```
        """)
