import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline
from wordcloud import WordCloud
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.dataframe_explorer import dataframe_explorer

# ------------------------------
# Utility functions for cleaning
# ------------------------------

def remove_html_tags(text):
    """Remove HTML tags using regex"""
    return re.sub(r'<.*?>', '', text)

stopwords = set(['a','an','the','and','or','is','are','was','were','in','on','at','of','to','for','it','how','can','i'])

def remove_stopwords(text):
    """Remove common stopwords from text"""
    words = re.findall(r'\b\w+\b', text.lower())
    filtered = [w for w in words if w not in stopwords]
    return ' '.join(filtered)

def clean_dataframe(df):
    """Clean dataframe columns and text"""
    if 'channel' in df.columns:
        df['channel'] = df['channel'].astype(str).str.lower()
    df['message'] = df['message'].astype(str).str.lower()
    df['category'] = df['category'].astype(str).str.lower()
    df['message'] = df['message'].apply(remove_html_tags)
    df['message'] = df['message'].apply(remove_stopwords)
    return df

# ------------------------------
# Model training and prediction
# ------------------------------

def train_model(df):
    """Train an SVM classifier on cleaned data"""
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['message'])
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    return model, vectorizer, report, matrix

def predict_category(query, model, vectorizer):
    """Predict category from a user query"""
    cleaned = remove_stopwords(remove_html_tags(query))
    X_query = vectorizer.transform([cleaned])
    prediction = model.predict(X_query)[0]
    probabilities = model.predict_proba(X_query)[0]
    return prediction, probabilities

# ------------------------------
# Text generation with transformers
# ------------------------------

@st.cache_resource(show_spinner=False)
def load_generator():
    """Load GPT2 text generation pipeline"""
    return pipeline("text-generation", model="distilgpt2")

def generate_response(query, generator):
    """Generate response text for a query"""
    prompt = f"Customer query: {query}. Response:"
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

# ------------------------------
# Visualization functions
# ------------------------------

def plot_wordcloud(text, title):
    """Generate and display word cloud"""
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap='viridis').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=16)
    st.pyplot(plt)

def plot_category_distribution(df):
    """Plot category distribution with enhanced styling"""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(y='category', data=df, 
                      order=df['category'].value_counts().index,
                      palette='viridis')
    plt.title("Message Distribution by Category", fontsize=16)
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Category", fontsize=12)
    
    # Add value labels
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 1, p.get_y() + p.get_height()/2.,
                '{:1.0f}'.format(width),
                ha='left', va='center', fontsize=10)
    
    st.pyplot(plt)

# ------------------------------
# Streamlit app main
# ------------------------------

# Set page config with favicon
st.set_page_config(
    page_title="Query Classification & Response App", 
    layout="wide",
    page_icon="💬"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 5px;
        padding: 0.5rem;
    }
    .stSelectbox>div>div>select {
        border-radius: 5px;
        padding: 0.5rem;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #e6f7ff;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #1890ff;
    }
</style>
""", unsafe_allow_html=True)

# Main title and description
st.title("💬 Query Classification and Auto Response Application")

st.markdown("""
<div style="background-color: #e6f7ff; padding: 1rem; border-radius: 10px; border-left: 4px solid #1890ff;">
    <p style="font-size: 16px; margin-bottom: 0;">
    This application helps you analyze customer queries, classify them into categories, and generate automated responses.
    Use the sidebar to navigate through the workflow steps.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation with icons
with st.sidebar:
    st.markdown("## 🔍 Navigation")
    page = st.radio("Choose a step:", 
        ["📂 Upload & Clean Data", 
         "📊 EDA & Visualization", 
         "🤖 Train Model", 
         "🧠 Classify Query", 
         "💡 Generate Response"])
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app uses:
    - SVM for text classification
    - TF-IDF for text vectorization
    - GPT-2 for response generation
    """)

# Session state initialization
if 'df_cleaned' not in st.session_state:
    st.session_state['df_cleaned'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'vectorizer' not in st.session_state:
    st.session_state['vectorizer'] = None
if 'generator' not in st.session_state:
    st.session_state['generator'] = load_generator()
if 'report' not in st.session_state:
    st.session_state['report'] = None

# ------------------------------
# Page 1: Upload & Clean Data
# ------------------------------

if page == "📂 Upload & Clean Data":
    st.header("📂 Upload & Clean Your Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV file with columns: 'message', 'category' (and optionally 'channel')", 
                                       type=["csv"], 
                                       help="The file should contain at least message and category columns")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("Raw Data Preview")
            filtered_df = dataframe_explorer(df)
            st.dataframe(filtered_df, use_container_width=True)
            
            if st.button("✨ Clean Data", help="Click to clean the data by removing HTML tags and stopwords"):
                try:
                    with st.spinner("Cleaning data..."):
                        df_cleaned = clean_dataframe(df)
                        st.session_state['df_cleaned'] = df_cleaned
                        st.success("Data cleaned successfully!")
                        
                        st.subheader("Cleaned Data Preview")
                        st.dataframe(df_cleaned.head(), use_container_width=True)
                        
                        # Show basic stats
                        st.subheader("Basic Statistics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Messages", len(df_cleaned))
                        col2.metric("Unique Categories", df_cleaned['category'].nunique())
                        if 'channel' in df_cleaned.columns:
                            col3.metric("Unique Channels", df_cleaned['channel'].nunique())
                        style_metric_cards()
                except Exception as e:
                    st.error(f"Error during cleaning: {e}")
    
    with col2:
        st.markdown("### Data Requirements")
        st.markdown("""
        <div class="highlight">
        Your CSV file should contain at least these columns:
        - <b>message</b>: The customer query text
        - <b>category</b>: The target classification label
        
        Optional column:
        - <b>channel</b>: Source of the message (email, chat, etc.)
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.get('df_cleaned') is not None:
            st.markdown("### Next Steps")
            st.markdown("""
            <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 5px;">
            Proceed to <b>EDA & Visualization</b> to explore your cleaned data.
            </div>
            """, unsafe_allow_html=True)

# ------------------------------
# Page 2: EDA & Visualization
# ------------------------------

elif page == "📊 EDA & Visualization":
    st.header("📊 Exploratory Data Analysis")
    
    df = st.session_state.get('df_cleaned')
    if df is None:
        st.warning("Please upload and clean data first in the 'Upload & Clean Data' section.")
    else:
        # Show metrics at the top
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Messages", len(df))
        col2.metric("Unique Categories", df['category'].nunique())
        if 'channel' in df.columns:
            col3.metric("Unique Channels", df['channel'].nunique())
        style_metric_cards()
        
        # Category distribution
        st.subheader("Category Distribution")
        plot_category_distribution(df)
        
        # Word cloud for all messages
        st.subheader("Word Cloud - All Messages")
        all_text = " ".join(df['message'].tolist())
        plot_wordcloud(all_text, "Most Frequent Words in All Messages")
        
        # Interactive exploration
        st.subheader("Interactive Data Exploration")
        selected_cat = st.selectbox("Select category to explore", 
                                  options=["All"] + list(df['category'].unique()))
        
        if selected_cat == "All":
            filtered_df = df
        else:
            filtered_df = df[df['category'] == selected_cat]
            
        st.dataframe(filtered_df.sample(min(10, len(filtered_df))[['message', 'category']], 
                     use_container_width=True)
        
        # Word cloud for selected category
        if selected_cat != "All":
            st.subheader(f"Word Cloud - {selected_cat} Messages")
            cat_text = " ".join(filtered_df['message'].tolist())
            plot_wordcloud(cat_text, f"Most Frequent Words in {selected_cat}")

# ------------------------------
# Page 3: Train Model
# ------------------------------

elif page == "🤖 Train Model":
    st.header("🤖 Train Classification Model")
    
    df = st.session_state.get('df_cleaned')
    if df is None:
        st.warning("Please upload and clean data first in the 'Upload & Clean Data' section.")
    else:
        st.markdown("""
        <div class="highlight">
        This will train an SVM classifier using TF-IDF vectorization on your cleaned data.
        The model will be saved in session for future predictions.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Train Model", help="Click to train the classification model"):
            with st.spinner("Training model (this may take a few minutes)..."):
                try:
                    model, vectorizer, report, matrix = train_model(df)
                    st.session_state['model'] = model
                    st.session_state['vectorizer'] = vectorizer
                    st.session_state['report'] = report
                    
                    st.success("Model trained successfully!")
                    st.balloons()
                    
                    # Display metrics
                    st.subheader("Model Performance")
                    accuracy = report['accuracy']
                    macro_avg = report['macro avg']
                    weighted_avg = report['weighted avg']
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{accuracy:.2%}")
                    col2.metric("Macro Avg F1", f"{macro_avg['f1-score']:.2%}")
                    col3.metric("Weighted Avg F1", f"{weighted_avg['f1-score']:.2%}")
                    style_metric_cards()
                    
                    # Classification report
                    st.subheader("Detailed Classification Report")
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.2f}", subset=pd.IndexSlice[:, ['precision', 'recall', 'f1-score']]), 
                                use_container_width=True)
                    
                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=df['category'].unique(),
                               yticklabels=df['category'].unique())
                    plt.title("Confusion Matrix", fontsize=16)
                    plt.xlabel("Predicted", fontsize=12)
                    plt.ylabel("Actual", fontsize=12)
                    st.pyplot(plt)
                    
                except Exception as e:
                    st.error(f"Error during model training: {e}")
        
        if st.session_state.get('model') is not None:
            st.markdown("### Next Steps")
            st.markdown("""
            <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 5px;">
            Your model is ready! Proceed to <b>Classify Query</b> to test it with new messages.
            </div>
            """, unsafe_allow_html=True)

# ------------------------------
# Page 4: Classify Query
# ------------------------------

elif page == "🧠 Classify Query":
    st.header("🧠 Query Category Prediction")
    
    model = st.session_state.get('model')
    vectorizer = st.session_state.get('vectorizer')
    
    if model is None or vectorizer is None:
        st.warning("Please train the model first in the 'Train Model' section.")
    else:
        st.markdown("""
        <div class="highlight">
        Enter a query below to classify it using your trained model. The system will show the predicted category and confidence scores.
        </div>
        """, unsafe_allow_html=True)
        
        query = st.text_area("Enter query text:", 
                           placeholder="Type your customer query here...",
                           height=150)
        
        if query:
            with st.spinner("Analyzing query..."):
                prediction, probabilities = predict_category(query, model, vectorizer)
                
                # Get all category names (classes)
                categories = model.classes_
                
                # Create a DataFrame for probabilities
                prob_df = pd.DataFrame({
                    'Category': categories,
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False)
                
                st.success(f"**Predicted Category:** {prediction}")
                
                # Show probability distribution
                st.subheader("Category Probabilities")
                
                # Bar chart of probabilities
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x='Probability', y='Category', data=prob_df, palette='viridis')
                plt.title("Prediction Confidence by Category", fontsize=16)
                plt.xlabel("Probability", fontsize=12)
                plt.ylabel("Category", fontsize=12)
                plt.xlim(0, 1)
                
                # Add value labels
                for p in ax.patches:
                    width = p.get_width()
                    plt.text(width + 0.01, p.get_y() + p.get_height()/2.,
                            '{:1.2f}'.format(width),
                            ha='left', va='center', fontsize=10)
                
                st.pyplot(plt)
                
                # Show top 3 categories in metrics
                st.subheader("Top Predictions")
                cols = st.columns(3)
                for i in range(min(3, len(prob_df))):
                    with cols[i]:
                        st.metric(label=f"#{i+1}: {prob_df.iloc[i]['Category']}", 
                                value=f"{prob_df.iloc[i]['Probability']:.2%}")

# ------------------------------
# Page 5: Generate Response
# ------------------------------

elif page == "💡 Generate Response":
    st.header("💡 Auto Response Generation")
    
    generator = st.session_state['generator']
    
    st.markdown("""
    <div class="highlight">
    This feature uses GPT-2 to generate suggested responses to customer queries. 
    Enter a query below to generate a response.
    </div>
    """, unsafe_allow_html=True)
    
    query = st.text_area("Enter query text:", 
                       placeholder="Type your customer query here...",
                       height=150,
                       key="response_query")
    
    if st.button("Generate Response", help="Click to generate a response using GPT-2"):
        if not query:
            st.warning("Please enter a query first.")
        else:
            with st.spinner("Generating response (this may take a few seconds)..."):
                try:
                    response = generate_response(query, generator)
                    # Clean up the response to remove the prompt
                    cleaned_response = response.replace(f"Customer query: {query}. Response:", "").strip()
                    
                    st.subheader("Generated Response")
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; border-left: 4px solid #28a745;">
                    {cleaned_response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show the full generated text in expander
                    with st.expander("View full generation"):
                        st.code(response)
                except Exception as e:
                    st.error(f"Error during response generation: {e}")
