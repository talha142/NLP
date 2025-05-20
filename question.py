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
import openai

# Set page config
st.set_page_config(page_title="Smart Query Assistant", layout="wide")
st.title("🤖 Smart Query Assistant")

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
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Model training function
def train_model(df):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, vectorizer, report, cm

# Prediction function
def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return pred, proba

# GPT response generation
def generate_gpt_response(query, category):
    prompt = f"""
    You are a customer service assistant. Respond to this {category} category query professionally and helpfully.
    
    Query: {query}
    
    Response:
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful customer service assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Could not generate response: {str(e)}"

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["📤 Upload Data", "🔍 Explore Data", "⚙️ Train Model", "💬 Classify & Respond"])

# Page 1: Upload Data
if page == "📤 Upload Data":
    st.header("Upload Your Query Data")
    uploaded_file = st.file_uploader("Choose a CSV file with 'message' and 'category' columns", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['message'] = df['message'].apply(clean_text)
        st.session_state.df = df
        st.success(f"Data loaded successfully! {len(df)} queries found.")
        st.dataframe(df.head(3))

# Page 2: Explore Data
elif page == "🔍 Explore Data":
    st.header("Explore Your Data")
    
    if st.session_state.df is None:
        st.warning("Please upload data first")
    else:
        df = st.session_state.df
        
        # Show basic stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Queries", len(df))
        col2.metric("Categories", df['category'].nunique())
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
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

# Page 3: Train Model
elif page == "⚙️ Train Model":
    st.header("Train Classification Model")
    
    if st.session_state.df is None:
        st.warning("Please upload data first")
    else:
        if st.button("Train Model"):
            with st.spinner("Training model (this may take a minute)..."):
                model, vectorizer, report, cm = train_model(st.session_state.df)
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                st.success("Model trained successfully!")
                
                # Show results
                st.subheader("Model Performance")
                st.write(f"Accuracy: {report['accuracy']:.2%}")
                
                st.subheader("Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0))
                
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=model.classes_, yticklabels=model.classes_)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig)

# Page 4: Classify & Respond
elif page == "💬 Classify & Respond":
    st.header("Classify & Generate Response")
    
    if st.session_state.model is None:
        st.warning("Please train the model first")
    else:
        query = st.text_area("Enter customer query:", height=150)
        
        if st.button("Analyze & Respond"):
            if not query:
                st.error("Please enter a query")
            else:
                with st.spinner("Analyzing query..."):
                    # Classify query
                    pred, proba = predict(query, st.session_state.model, st.session_state.vectorizer)
                    
                    # Show classification results
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
                    
                    # Generate GPT response
                    with st.spinner("Generating response..."):
                        st.subheader("Suggested Response")
                        response = generate_gpt_response(query, pred)
                        st.markdown(f"""
                        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; border-left: 4px solid #1e90ff;">
                        {response}
                        </div>
                        """, unsafe_allow_html=True)

# Add OpenAI API key input
st.sidebar.markdown("---")
st.sidebar.subheader("GPT Settings")
openai.api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if openai.api_key:
    st.sidebar.success("API key set!")
else:
    st.sidebar.warning("Enter API key for GPT responses")
