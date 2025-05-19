# ===== app.py =====
import streamlit as st
from streamlit_option_menu import option_menu

# Configure page
st.set_page_config(page_title="Customer Support NLP App", layout="wide")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home", "EDA", "Predict & Answer"],
        icons=["house", "bar-chart", "robot"],
        menu_icon="cast",
        default_index=0
    )

# ===== HOME PAGE =====
if selected == "Home":
    st.title("📞 Customer Support NLP App")
    st.markdown("""
    Welcome to the multi-page NLP-powered customer support app.

    ### What You Can Do:
    - 📊 Explore your customer support data.
    - 🤖 Predict the category of customer queries.
    - 💬 Automatically generate smart responses.

    **Get started by choosing a page from the sidebar.**
    """)

# ===== EDA PAGE =====
elif selected == "EDA":
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import re

    st.title("📊 Step 1: EDA - Explore Your Data")
    uploaded_file = st.file_uploader("Upload your customer support CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        st.subheader("Missing Values Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        st.pyplot(fig)

        st.subheader("Value Counts by Category")
        if 'category' in df.columns:
            st.bar_chart(df['category'].value_counts())

        # Preprocess and cache for next step
        for col in ['channel', 'message', 'category']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower()

        def remove_html_tags(text):
            return re.sub(r'<.*?>', '', text)

        stopwords = set(['a','an','the','and','or','is','are','was','were','in','on','at','of','to','for','it','how','can','i'])
        def remove_stopwords(text):
            words = re.findall(r'\b\w+\b', text.lower())
            return ' '.join([w for w in words if w not in stopwords])

        df['message'] = df['message'].apply(remove_html_tags).apply(remove_stopwords)

        st.session_state.cleaned_data = df
        st.success("Data cleaned and stored for modeling!")
    else:
        st.info("Please upload a file to begin.")

# ===== PREDICT PAGE =====
elif selected == "Predict & Answer":
    import pandas as pd
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    from transformers import pipeline

    st.title("🤖 Step 2: Predict Category & Generate Answers")

    if 'cleaned_data' in st.session_state:
        df = st.session_state.cleaned_data

        # Vectorize and train model
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['message'])
        y = df['category']

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)

        st.subheader("Model Evaluation")
        st.text(classification_report(y_test, model.predict(X_test)))

        query = st.text_area("✉️ Enter a customer query to classify and generate a response:")
        if st.button("Predict and Generate"):
            def clean_text(text):
                text = re.sub(r'<.*?>', '', text)
                text = re.sub(r'[^a-zA-Z ]', '', text)
                return ' '.join([w for w in text.split() if w.lower() not in stopwords])

            cleaned_query = clean_text(query)
            X_input = vectorizer.transform([cleaned_query])
            predicted_cat = model.predict(X_input)[0]

            st.info(f"Predicted Category: {predicted_cat}")

            @st.cache_resource
            def load_generator():
                return pipeline("text-generation", model="distilgpt2")
            generator = load_generator()

            prompt = f"Category: {predicted_cat}\nCustomer Query: {query}\nResponse:"
            result = generator(prompt, max_length=100, num_return_sequences=1)

            st.success("Generated Response:")
            st.write(result[0]['generated_text'])
    else:
        st.warning("⚠️ Please upload and clean your data on the EDA page first.")
