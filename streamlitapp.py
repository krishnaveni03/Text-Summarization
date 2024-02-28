import streamlit as st
from summarizer.sbert import SBertSummarizer
from sentence_transformers import SentenceTransformer

# Load SBERT model for encoding
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to generate summary using SBERT
def generate_summary_sbert(body):
    summary = sbert_model.encode(body, convert_to_tensor=False)
    return summary[0]  # Return the first element of the summary

# Function to generate summary using SBertSummarizer
def generate_summary_sbertsummarizer(body):
    model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
    summary = model(body, num_sentences=5)
    return summary

# Home page
def home():
    st.title("Welcome to Text Summarization")
    st.write("""
    This is a simple Streamlit app that demonstrates text summarization using SBERT (Sentence-BERT) models.
    Choose the 'Predict' page from the sidebar to enter text and generate a summary.
    """)

# Predict page
def predict():
    st.title("Text Summarization")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "Predict"))

    if page == "Home":
        home()
    elif page == "Predict":
        st.subheader("Generate Summary")
        body = st.text_area("Enter the text to summarize:")
        if st.button("Generate Summary"):
            if body:
                # Generate summary using SBERT
                summary_sbert = generate_summary_sbert(body)
                st.subheader("Summary (SBERT):")
                st.write(summary_sbert)

                # Generate summary using SBertSummarizer
                summary_sbertsummarizer = generate_summary_sbertsummarizer(body)
                st.subheader("Summary (SBertSummarizer):")
                st.write(summary_sbertsummarizer)
            else:
                st.warning("Please enter some text to summarize.")

def main():
    predict()

if __name__ == "__main__":
    main()
