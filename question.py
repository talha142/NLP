# main.py
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Customer Query Classifier", layout="wide")

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Upload & Clean", "EDA", "Train Model", "Query & Help"])

if page == "Upload & Clean":
    st.title("📤 Step 1: Upload & Clean Your Data")
    st.write("Upload a CSV file containing customer queries and their categories.")
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"])

    if uploaded_file:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.write("Raw Data:", df.head())

        # Simple Cleaning
        df = df.dropna(subset=["message", "category"])
        df['message'] = df['message'].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)

        st.session_state.cleaned_data = df
        st.success("Cleaned and saved to session_state.cleaned_data")

elif page == "EDA":
    import seaborn as sns
    import matplotlib.pyplot as plt

    st.title("📊 Step 2: Exploratory Data Analysis")

    if 'cleaned_data' in st.session_state:
        df = st.session_state.cleaned_data
        st.subheader("Class Distribution")
        st.bar_chart(df['category'].value_counts())

        st.subheader("Heatmap of Category vs Word Count")
        df['length'] = df['message'].str.split().apply(len)
        pivot = df.pivot_table(index='category', values='length', aggfunc='mean')
        fig, ax = plt.subplots()
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please upload and clean data first.")

elif page == "Train Model":
    st.title("🤖 Step 3: Fine-tune DistilBERT for Classification")
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    import torch

    if 'cleaned_data' in st.session_state:
        df = st.session_state.cleaned_data
        label_enc = LabelEncoder()
        df['label'] = label_enc.fit_transform(df['category'])
        hf_dataset = Dataset.from_pandas(df[['message', 'label']])

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        def tokenize(batch):
            return tokenizer(batch['message'], padding=True, truncation=True)

        hf_dataset = hf_dataset.map(tokenize, batched=True)
        hf_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        split = hf_dataset.train_test_split(test_size=0.2)
        train_ds = split['train']
        test_ds = split['test']

        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_enc.classes_))

        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_dir='./logs'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds
        )

        trainer.train()

        predictions = trainer.predict(test_ds)
        preds = predictions.predictions.argmax(axis=1)
        true_labels = predictions.label_ids

        st.subheader("Classification Report")
        st.text(classification_report(true_labels, preds, target_names=label_enc.classes_))
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(true_labels, preds))

        st.session_state.label_enc = label_enc
        st.session_state.bert_model = model
        st.session_state.tokenizer = tokenizer
    else:
        st.warning("Please upload and preprocess data first.")

elif page == "Query & Help":
    st.title("🗣️ Step 4: Query Prediction & Help Center")

    if all(k in st.session_state for k in ['bert_model', 'tokenizer', 'label_enc']):
        model = st.session_state.bert_model
        tokenizer = st.session_state.tokenizer
        label_enc = st.session_state.label_enc

        st.subheader("🔍 Predict Category for Customer Query")
        query = st.text_area("Enter a customer query:")
        if st.button("Classify"):
            inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted = torch.argmax(logits, dim=1).item()
                label = label_enc.inverse_transform([predicted])[0]
            st.success(f"Predicted Category: **{label}**")

        st.subheader("💬 Help Center")
        st.markdown("""
        - **Having Trouble?** Make sure you've trained the model first.
        - **Want Better Accuracy?** Try cleaning data and increasing training epochs.
        - **Data Format Issue?** Input CSV must contain `message` and `category` columns.
        - **Need Help?** Contact your dev team or refer to [HuggingFace Docs](https://huggingface.co/docs).
        """)
    else:
        st.warning("Please complete training before classifying queries.")
