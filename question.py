import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline
import plotly.express as px
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set up the page
st.set_page_config(page_title="AI Customer Support Assistant", layout="wide", page_icon="🤖")
st.title("🤖 AI Customer Support Assistant")
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stTextArea textarea {font-size: 16px;}
    .stButton button {width: 100%; background-color: #4CAF50; color: white;}
    .stButton button:hover {background-color: #45a049;}
    .stAlert {border-left: 5px solid #4CAF50;}
    </style>
    """, unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1570/1570887.png", width=100)
    st.title("Navigation")
    app_mode = st.radio("Choose a mode:", 
                       ["Data Analysis", "Model Training", "Live Prediction", "Response Generation"])
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app uses NLP to classify customer queries and generate responses.")
    
# File uploader
uploaded_file = st.file_uploader("📤 Upload your customer support dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Data cleaning functions
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text
    
    # Apply cleaning
    df['cleaned_message'] = df['message'].apply(clean_text)
    
    if app_mode == "Data Analysis":
        st.header("📊 Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.dataframe(df.head())
            st.metric("Total Samples", len(df))
            
            # Basic stats
            st.subheader("Basic Statistics")
            st.write(f"Number of unique categories: {df['category'].nunique()}")
            st.write(f"Number of unique channels: {df['channel'].nunique()}")
            
        with col2:
            st.subheader("Category Distribution")
            fig = px.pie(df, names='category', title='Query Categories')
            st.plotly_chart(fig, use_container_width=True)
            
        # Word cloud
        st.subheader("Word Cloud of Customer Queries")
        text = " ".join(review for review in df.cleaned_message)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        
        # Channel analysis
        st.subheader("Channel Analysis")
        channel_counts = df['channel'].value_counts().reset_index()
        channel_counts.columns = ['Channel', 'Count']
        fig = px.bar(channel_counts, x='Channel', y='Count', color='Channel')
        st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "Model Training":
        st.header("🤖 Model Training")
        
        # Encode labels
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['category'])
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['cleaned_message'])
        y = df['category_encoded']
        
        # Train-test split
        test_size = st.slider("Select test set size", 0.1, 0.5, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42)
        
        # Model training
        st.subheader("Train Classification Model")
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                start_time = time.time()
                
                model = SVC(kernel='linear', probability=True)
                model.fit(X_train, y_train)
                
                training_time = time.time() - start_time
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = np.mean(y_pred == y_test)
                
                st.success(f"Model trained successfully in {training_time:.2f} seconds!")
                st.metric("Accuracy", f"{accuracy*100:.2f}%")
                
                # Classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, 
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=le.classes_, 
                               y=le.classes_,
                               text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "Live Prediction":
        st.header("🔍 Live Query Classification")
        
        if 'model' not in st.session_state:
            st.warning("Please train the model first in the 'Model Training' section.")
        else:
            query = st.text_area("Enter a customer query:", 
                               placeholder="Type your customer support query here...",
                               height=150)
            
            if st.button("Classify Query"):
                if query.strip():
                    # Clean and vectorize query
                    cleaned_query = clean_text(query)
                    X_query = st.session_state.vectorizer.transform([cleaned_query])
                    
                    # Predict
                    predicted_num = st.session_state.model.predict(X_query)[0]
                    predicted_category = st.session_state.le.inverse_transform([predicted_num])[0]
                    probabilities = st.session_state.model.predict_proba(X_query)[0]
                    
                    # Display results
                    st.subheader("Classification Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Category", predicted_category)
                        
                    with col2:
                        max_prob = np.max(probabilities) * 100
                        st.metric("Confidence", f"{max_prob:.1f}%")
                    
                    # Show probabilities
                    st.subheader("Category Probabilities")
                    prob_df = pd.DataFrame({
                        'Category': st.session_state.le.classes_,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    
                    fig = px.bar(prob_df, x='Probability', y='Category', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show similar past queries
                    st.subheader("Similar Past Queries")
                    similar_queries = df[df['category'] == predicted_category].sample(3)
                    for i, row in similar_queries.iterrows():
                        st.markdown(f"**{row['category']}**: {row['message']}")
                else:
                    st.warning("Please enter a query to classify.")
    
    elif app_mode == "Response Generation":
        st.header("💬 AI Response Generator")
        
        # Load text generation model
        @st.cache_resource
        def load_generator():
            return pipeline("text-generation", model="gpt2")
        
        generator = load_generator()
        
        query = st.text_area("Enter customer query for response generation:",
                            placeholder="The product I received is damaged...",
                            height=150)
        
        col1, col2 = st.columns(2)
        with col1:
            max_length = st.slider("Response length", 50, 300, 150)
        with col2:
            temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
        
        if st.button("Generate Response"):
            if query.strip():
                with st.spinner("Generating response..."):
                    prompt = f"Customer support query: {query}\n\nAI response:"
                    response = generator(
                        prompt,
                        max_length=max_length,
                        num_return_sequences=1,
                        temperature=temperature,
                        do_sample=True
                    )
                    
                    generated_text = response[0]['generated_text']
                    # Extract just the response part
                    ai_response = generated_text.split("AI response:")[1].strip()
                    
                    st.subheader("Generated Response")
                    st.success(ai_response)
                    
                    # Feedback mechanism
                    st.subheader("Feedback")
                    feedback = st.radio("Was this response helpful?", 
                                      ("👍 Yes", "👎 No"), 
                                      horizontal=True)
                    if feedback:
                        st.write("Thank you for your feedback!")
            else:
                st.warning("Please enter a query to generate a response.")
else:
    st.info("👈 Please upload a CSV file to begin. The file should contain 'message' and 'category' columns.")
