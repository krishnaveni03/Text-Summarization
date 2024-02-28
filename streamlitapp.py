import streamlit as st
from summarizer.sbert import SBertSummarizer
from sentence_transformers import SentenceTransformer
import os

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

# Function to download summary as a text file
def download_summary(summary_text):
    with open("summary.txt", "w") as f:
        f.write(summary_text)
    st.success("Summary has been downloaded as 'summary.txt'")

# Home page
def home():
    st.title("Welcome to Text Summarization")
    st.write("""
    This is a simple Streamlit app that demonstrates text summarization using SBERT (Sentence-BERT) models.
    Choose the 'Predict' page from the sidebar to enter text and generate a summary.
    """)

    st.write("## Upload a Text File")
    uploaded_file = st.file_uploader("Choose a file", type=['txt'])
    if uploaded_file is not None:
        file_contents = uploaded_file.read()
        st.write("## File Content:")
        st.write(file_contents)

        if st.button("Generate Summary"):
            summary_sbertsummarizer = generate_summary_sbertsummarizer(file_contents)
            st.subheader("Summary (SBertSummarizer):")
            st.write(summary_sbertsummarizer)

            if st.button("Download Summary"):
                download_summary(summary_sbertsummarizer.decode("utf-8"))

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
                summary_sbertsummarizer = generate_summary_sbertsummarizer(body)
                st.subheader("Summary (SBertSummarizer):")
                st.write(summary_sbertsummarizer)
            else:
                st.warning("Please enter some text to summarize.")

def main():
    predict()

if __name__ == "__main__":
    main()
