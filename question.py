import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from PIL import Image

# Configure page
st.set_page_config(
    page_title="Customer Support Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --primary: #3498db;
        --secondary: #2ecc71;
        --dark: #2c3e50;
        --light: #ecf0f1;
    }
    .sidebar .sidebar-content {
        background-color: var(--dark);
        color: white;
    }
    .sidebar .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px 0;
        width: 100%;
    }
    .sidebar .stButton>button:hover {
        background-color: #2980b9;
    }
    .page-container {
        background-color: white;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .data-preview {
        border: 2px solid #e1e4e8;
        border-radius: 8px;
        padding: 15px;
    }
    .clean-highlight {
        background-color: #e6ffed;
        padding: 2px 4px;
        border-radius: 4px;
    }
    .original-highlight {
        background-color: #ffebe9;
        padding: 2px 4px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
# SIDEBAR NAVIGATION
# ==============================================
with st.sidebar:
    st.image("https://via.placeholder.com/200x50?text=Support+Pro", width=200)
    st.markdown("<h2 style='color: white;'>Navigation</h2>", unsafe_allow_html=True)
    
    # Page selection buttons
    page = st.radio(
        "Choose a page:",
        ["🏠 Dashboard", "📊 Data Preview", "🧹 Data Cleaning", "📈 Advanced EDA"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # File uploader in sidebar
    uploaded_file = st.file_uploader(
        "📤 Upload Dataset", 
        type=["csv", "xlsx"],
        help="Supports CSV and Excel files"
    )
    
    st.markdown("---")
    st.markdown("<p style='color: var(--light);'>Version 2.0</p>", unsafe_allow_html=True)

# Load data (shared across pages)
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Basic cleaning (applied to all pages)
    df_clean = df.copy()
    df_clean['message'] = df_clean['message'].astype(str).str.lower()
    df_clean['category'] = df_clean['category'].astype(str).str.lower().str.strip()
    df_clean = df_clean.dropna(subset=['message', 'category'])

# ==============================================
# PAGE 1: DASHBOARD
# ==============================================
if page == "🏠 Dashboard":
    st.title("📊 Customer Support Analytics Dashboard")
    
    if uploaded_file:
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", len(df_clean))
        with col2:
            st.metric("Unique Categories", df_clean['category'].nunique())
        with col3:
            st.metric("Avg Message Length", f"{df_clean['message'].apply(len).mean():.0f} chars")
        
        # Quick preview
        with st.container():
            st.subheader("Quick Data Preview")
            st.dataframe(df_clean.head(3))
            
        # Sample visualization
        with st.container():
            st.subheader("Category Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            df_clean['category'].value_counts().head(5).plot(kind='bar', ax=ax, color='#3498db')
            ax.set_title("Top 5 Categories")
            st.pyplot(fig)
    else:
        st.info("Please upload a dataset to begin analysis")
        st.image("https://via.placeholder.com/800x400?text=Upload+Your+Customer+Support+Data", use_column_width=True)

# ==============================================
# PAGE 2: DATA PREVIEW
# ==============================================
elif page == "📊 Data Preview":
    st.title("🔍 Data Preview")
    
    if uploaded_file:
        # Data summary
        with st.container():
            st.subheader("Dataset Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**First 5 rows:**")
                st.dataframe(df.head())
            with col2:
                st.write("**Data Summary:**")
                st.write(f"Shape: {df.shape}")
                st.write(f"Columns: {list(df.columns)}")
                st.write(f"Missing Values: {df.isnull().sum().sum()}")
        
        # Data statistics
        with st.container():
            st.subheader("Detailed Statistics")
            tab1, tab2 = st.tabs(["Categorical", "Numerical"])
            with tab1:
                st.write(df.describe(include=['object']))
            with tab2:
                st.write(df.describe(include=['number']))
    else:
        st.warning("No dataset uploaded")

# ==============================================
# PAGE 3: DATA CLEANING
# ==============================================
elif page == "🧹 Data Cleaning":
    st.title("✨ Data Cleaning Studio")
    
    if uploaded_file:
        # Cleaning options
        with st.expander("⚙️ Cleaning Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                remove_html = st.checkbox("Remove HTML tags", True)
                remove_stopwords = st.checkbox("Remove stop words", True)
            with col2:
                clean_emails = st.checkbox("Remove emails", True)
                clean_urls = st.checkbox("Remove URLs", True)
        
        # Cleaning function
        def clean_text(text):
            original = str(text)
            cleaned = original.lower()
            
            if remove_html:
                cleaned = re.sub(r'<.*?>', '', cleaned)
            if clean_urls:
                cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned)
            if clean_emails:
                cleaned = re.sub(r'\S+@\S+', '', cleaned)
            if remove_stopwords:
                stopwords = ['a','an','the','and','or','is','are','was','were']
                cleaned = ' '.join([word for word in cleaned.split() if word not in stopwords])
            
            return original, cleaned
        
        # Interactive preview
        st.subheader("Interactive Cleaning Preview")
        sample_size = st.slider("Sample size", 1, 10, 3)
        
        for i in range(sample_size):
            original, cleaned = clean_text(df.iloc[i]['message'])
            st.markdown(f"""
            <div class='data-preview'>
                <h4>Example {i+1}</h4>
                <p><strong>Original:</strong> <span class='original-highlight'>{original}</span></p>
                <p><strong>Cleaned:</strong> <span class='clean-highlight'>{cleaned}</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Full dataset processing
        if st.button("Apply Cleaning to Full Dataset", type="primary"):
            with st.spinner("Processing full dataset..."):
                df_clean['message'] = df_clean['message'].apply(lambda x: clean_text(x)[1])
                st.success("Cleaning completed!")
                st.dataframe(df_clean.head())
    else:
        st.warning("Upload a dataset to use cleaning tools")

# ==============================================
# PAGE 4: ADVANCED EDA
# ==============================================
elif page == "📈 Advanced EDA":
    st.title("📊 Advanced Exploratory Analysis")
    
    if uploaded_file:
        # Visualization options
        with st.expander("Visualization Settings"):
            top_n = st.slider("Number of top categories", 5, 20, 10)
            color_palette = st.selectbox("Color palette", ["viridis", "plasma", "magma", "coolwarm"])
        
        # Tabbed interface
        tab1, tab2, tab3 = st.tabs(["Categories", "Text Analysis", "Correlations"])
        
        with tab1:
            st.subheader("Category Distribution")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            df_clean['category'].value_counts().head(top_n).plot(
                kind='barh', 
                ax=ax1, 
                color=sns.color_palette(color_palette, top_n)
            )
            ax1.set_title(f"Top {top_n} Categories")
            st.pyplot(fig1)
        
        with tab2:
            st.subheader("Text Characteristics")
            
            df_clean['message_length'] = df_clean['message'].apply(len)
            df_clean['word_count'] = df_clean['message'].apply(lambda x: len(x.split()))
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Message Length Distribution**")
                fig2, ax2 = plt.subplots()
                sns.histplot(df_clean['message_length'], kde=True, ax=ax2)
                st.pyplot(fig2)
            
            with col2:
                st.write("**Word Cloud**")
                wordcloud = WordCloud(width=600, height=300, background_color='white').generate(" ".join(df_clean['message']))
                fig3, ax3 = plt.subplots()
                ax3.imshow(wordcloud)
                ax3.axis("off")
                st.pyplot(fig3)
        
        with tab3:
            st.subheader("Feature Correlations")
            
            # Create numerical features
            df_clean['avg_word_length'] = df_clean['message'].apply(
                lambda x: np.mean([len(w) for w in x.split()])
            )
            
            # Correlation matrix
            corr_matrix = df_clean[['message_length', 'word_count', 'avg_word_length']].corr()
            
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap=color_palette, center=0, ax=ax4)
            ax4.set_title("Feature Correlation Heatmap")
            st.pyplot(fig4)
    else:
        st.warning("Upload a dataset to view analytics")

# ==============================================
# FOOTER
# ==============================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #7f8c8d;'>Customer Support Analyzer Pro © 2023</div>", 
            unsafe_allow_html=True)
