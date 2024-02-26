import streamlit as st
from summarizer.sbert import SBertSummarizer

# Function to generate summary
def generate_summary(body):
    model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
    summary = model(body, num_sentences=5)
    return summary

def home():
    st.title("Welcome to Text Summarization with SBERT")
    st.write("""
    This is a simple Streamlit app that demonstrates text summarization using SBERT (Sentence-BERT) model.
    Choose the 'Predict' page from the sidebar to enter text and generate a summary.
    """)

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
                summary = generate_summary(body)
                st.subheader("Summary:")
                st.write(summary)
            else:
                st.warning("Please enter some text to summarize.")

def main():
    predict()

if __name__ == "__main__":
    main()
