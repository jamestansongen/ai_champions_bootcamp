# Import relevant libraries
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Import function to check password
from helper_functions.utility import check_password

# Import function to process documents
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
    page_title="DocuMind: From pages to insights"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("DocuMind: From pages to insights")

# Process static PDFs on first load
if 'existing_docs_loaded' not in st.session_state:
    with st.spinner("Processing initial documents...this may take a few minutes"):
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

user_prompt = st.text_area("Your question (e.g. Tell me about Singapore's urban development/Give me examples where HDB was involved in Singapore's urban development/Which years were URA master plan mentioned?)", height=150)

if st.button("Submit Question") and user_prompt.strip():
    with st.spinner("Generating answer..."):
        response = answer_query_with_llm_filter(user_prompt)
        st.write(response)