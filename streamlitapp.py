import streamlit as st
from sentence_transformers import SentenceTransformer

# Load SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to generate summary
def generate_summary(body):
    # Perform summarization using the SBERT model
    # This is just a placeholder; actual implementation may vary
    summary = model.encode(body, convert_to_tensor=True)
    return summary

def main():
    st.title("Text Summarization with SBERT")
    body = st.text_area("Enter the text to summarize:")
    if st.button("Generate Summary"):
        if body:
            summary = generate_summary(body)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
