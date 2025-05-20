import streamlit as st
from backend import *  # Your existing backend functions

st.set_page_config(page_title="Customer Support Assistant", layout="wide")

def welcome_page():
    st.title("Customer Support Assistant")
    st.write("""
    This application helps categorize customer queries and generate responses.
    Features:
    - Upload your customer support data
    - Explore data through EDA
    - Train a classification model
    - Predict query categories
    - Generate automated responses
    """)


def eda_section():
    st.header("Data Exploration")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        cleaned_df = clean_dataframe(df)
        
        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned_df.head())
        
        # Add visualizations
        st.subheader("Category Distribution")
        st.bar_chart(cleaned_df['category'].value_counts())

def model_training_section(df):
    st.header("Model Training")
    
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            model, vectorizer, report, matrix = train_model(df)
            
            st.subheader("Model Performance")
            st.text(report)
            
            # Store model in session state
            st.session_state['model'] = model
            st.session_state['vectorizer'] = vectorizer

def prediction_section():
    st.header("Query Prediction")
    
    query = st.text_area("Enter your customer query:")
    
    if st.button("Predict and Generate Response"):
        if 'model' not in st.session_state:
            st.warning("Please train the model first")
        else:
            # Predict category
            category = predict_category(
                query,
                st.session_state['model'],
                st.session_state['vectorizer']
            )
            st.success(f"Predicted Category: {category}")
            
            # Generate response
            generator = load_generator()
            response = generate_response(query, generator)
            st.subheader("Generated Response")
            st.write(response.split("Response:")[1].strip())

def main():
    pages = {
        "Welcome": welcome_page,
        "EDA": eda_section,
        "Model Training": model_training_section,
        "Prediction": prediction_section
    }
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Execute selected page function
    if selection == "Model Training":
        if 'df' in st.session_state:
            pages[selection](st.session_state.df)
        else:
            st.warning("Please upload data first from EDA section")
    else:
        pages[selection]()

if __name__ == "__main__":
    main()
