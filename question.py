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

# Set page config
st.set_page_config(page_title="Query Classifier", layout="wide")
st.title("📩 Query Classification App")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None

# Text cleaning functions
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.lower()  # Convert to lowercase
    return text

# Model training function
def train_model(df):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['message'])
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, vectorizer, report, cm

# Prediction function
def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return pred, proba

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Upload Data", "Explore Data", "Train Model", "Classify Query"])

# Page 1: Upload Data
if page == "Upload Data":
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['message'] = df['message'].apply(clean_text)
        st.session_state.df = df
        st.success("Data loaded successfully!")
        st.dataframe(df.head())

# Page 2: Explore Data
elif page == "Explore Data":
    st.header("Explore Your Data")
    
    if st.session_state.df is None:
        st.warning("Please upload data first")
    else:
        df = st.session_state.df
        
        # Show basic stats
        col1, col2 = st.columns(2)
        col1.metric("Total Queries", len(df))
        col2.metric("Categories", df['category'].nunique())
        
        # Category distribution
        st.subheader("Category Distribution")
        fig, ax = plt.subplots()
        df['category'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
        
        # Word cloud
        st.subheader("Word Cloud")
        text = " ".join(df['message'].tolist())
        wordcloud = WordCloud(width=800, height=400).generate(text)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

# Page 3: Train Model
elif page == "Train Model":
    st.header("Train Classification Model")
    
    if st.session_state.df is None:
        st.warning("Please upload data first")
    else:
        if st.button("Train Model"):
            with st.spinner("Training..."):
                model, vectorizer, report, cm = train_model(st.session_state.df)
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                st.success("Model trained successfully!")
                
                # Show results
                st.subheader("Model Performance")
                st.text(report)
                
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                st.pyplot(fig)

# Page 4: Classify Query
elif page == "Classify Query":
    st.header("Classify New Query")
    
    if st.session_state.model is None:
        st.warning("Please train the model first")
    else:
        query = st.text_area("Enter your query:")
        
        if query:
            pred, proba = predict(query, st.session_state.model, st.session_state.vectorizer)
            st.success(f"Predicted category: {pred}")
            
            # Show probabilities
            st.subheader("Category Probabilities")
            prob_df = pd.DataFrame({
                'Category': st.session_state.model.classes_,
                'Probability': proba
            }).sort_values('Probability', ascending=False)
            
            fig, ax = plt.subplots()
            sns.barplot(x='Probability', y='Category', data=prob_df, ax=ax)
            st.pyplot(fig)
