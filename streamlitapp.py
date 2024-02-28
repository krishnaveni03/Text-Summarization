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

# Function to download summary as a text file
def download_summary(summary_text):
    with open("summary.txt", "w") as f:
        f.write(summary_text)
    st.success("Summary has been downloaded as 'summary.txt'")

# CSS styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Home page
def home():
    st.title("Welcome to Text Summarization")
    st.write("""
    This is a simple Streamlit app that demonstrates text summarization using SBERT (Sentence-BERT) models.
    Choose the 'Predict' page from the sidebar to enter text and generate a summary.
    """)

# Predict page
def predict():
    local_css("style.css")  # Load local CSS file
    st.title("Text Summarization")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "Predict"))

    if page == "Predict":
        st.subheader("Generate Summary")
        option = st.radio("Choose option", ("Enter Text", "Upload Text File"))

        if option == "Enter Text":
            body = st.text_area("Enter the text to summarize:")
        elif option == "Upload Text File":
            uploaded_file = st.file_uploader("Choose a file", type=['txt'])
            if uploaded_file is not None:
                body = uploaded_file.read().decode("utf-8")
            else:
                body = ""

        if st.button("Generate Summary"):
            if body:
                summary_sbertsummarizer = generate_summary_sbertsummarizer(body)
                st.subheader("Summary (SBertSummarizer):")
                st.write(summary_sbertsummarizer)

                if st.button("Download Summary"):
                    download_summary(summary_sbertsummarizer.decode("utf-8"))
            else:
                st.warning("Please enter some text to summarize.")

# Main function
def main():
    predict()

if __name__ == "__main__":
    main()
