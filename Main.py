# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# From helper_functions import check_password
from helper_functions.utility import check_password

# From helper functions import handle_document_upload
from logics.rag_pipeline import process_existing_documents, process_uploaded_document, answer_query_with_llm_filter

# Load environment variables from .env
load_dotenv()

# Check if the password is correct.  
if not check_password():  
    st.stop()

if 'doc_uploaded' not in st.session_state:
    st.session_state['doc_uploaded'] = False

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("Streamlit App")

# Process static PDFs on first load
if 'existing_docs_loaded' not in st.session_state:
    with st.spinner("Processing initial documents..."):
        try:
            process_existing_documents(folder_path="data")
            st.success("Static documents processed and vector store created.")
            st.session_state['existing_docs_loaded'] = True
        except Exception as e:
            st.error(f"Failed to process static documents: {e}")
            st.stop()

# Allow users to upload additional PDFs
uploaded_file = st.file_uploader("Upload additional PDF", type=["pdf"])
if uploaded_file and st.button("Add to vector store"):
    with st.spinner("Processing uploaded document..."):
        try:
            process_uploaded_document(uploaded_file)
            st.success("Uploaded document added to vector store.")
        except Exception as e:
            st.error(f"Failed to process uploaded document: {e}")

st.divider()
st.subheader("Ask a question")

user_prompt = st.text_area("Your question", height=150)

if st.button("Submit Question") and user_prompt.strip():
    with st.spinner("Generating answer..."):
        response = answer_query_with_llm_filter(user_prompt)
        st.write(response)
