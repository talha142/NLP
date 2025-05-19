# Import necessary libraries
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline

# Set page config
st.set_page_config(page_title="Customer Support NLP App", layout="wide")
st.title("🤖 Customer Support NLP App")

# Sidebar: Upload file + AI assistant
with st.sidebar:
    st.header("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Upload customer support data", type="csv")

    st.markdown("---")
    st.header("🧠 AI Assistant (Bard-like Bot)")
    user_question = st.text_area("Ask the AI Assistant:")
    if st.button("Get AI Answer"):
        st.info("AI bot functionality will be added soon (Gemini or ChatGPT API).")

# Tabs for layout
tab1, tab2, tab3 = st.tabs(["📊 Data View", "📈 Model Training", "💬 Classify & Respond"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Convert to lowercase
    df['channel'] = df['channel'].astype(str).str.lower()
    df['message'] = df['message'].astype(str).str.lower()
    df['category'] = df['category'].astype(str).str.lower()

    # Clean functions
    def remove_html_tags(text):
        return re.sub(r'<.*?>', '', text)

    stopwords = ['a','an','the','and','or','is','are','was','were','in','on','at','of','to','for','it','how','can','i']
    def remove_stopwords(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return ' '.join([w for w in words if w not in stopwords])

    # Apply cleaning
    df['message'] = df['message'].apply(remove_html_tags)
    df['message'] = df['message'].apply(remove_stopwords)

    # TF-IDF + Labels
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['message'])
    y = df['category']

    # Split & Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Hugging Face response model
    @st.cache_resource
    def load_generator():
        return pipeline("text-generation", model="distilgpt2")
    generator = load_generator()

    # ----- Tab 1 -----
    with tab1:
        st.subheader("📄 Raw & Cleaned Data")
        st.write("**First 5 Rows (Cleaned):**")
        st.dataframe(df.head())

    # ----- Tab 2 -----
    with tab2:
        st.subheader("📈 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("🔢 Confusion Matrix")
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

    # ----- Tab 3 -----
    with tab3:
        st.subheader("💬 Query Classifier & Auto-Responder")

        query = st.text_area("Enter a customer query to classify and generate a response:")

        if st.button("Classify Query & Generate Response"):
            if query.strip():
                cleaned_query = remove_html_tags(query)
                cleaned_query = remove_stopwords(cleaned_query)
                X_query = vectorizer.transform([cleaned_query])

                predicted_category = model.predict(X_query)[0]
                st.write(f"**Predicted Category:** `{predicted_category}`")

                # Generate response
                prompt = f"Customer query: {query}. Response:"
                response = generator(prompt, max_length=100, num_return_sequences=1)
                st.write("**Generated Response:**")
                st.success(response[0]['generated_text'])
            else:
                st.warning("Please enter a query first.")
else:
    with tab1:
        st.info("Upload a CSV file from the sidebar to view and process data.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | © 2025")
