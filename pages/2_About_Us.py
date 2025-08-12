import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="About Us",
    layout="centered"
)

# Page title
st.title("About Us")

# --- Project Scope ---
st.header("Project Scope")
st.write(
    """
    The project delivers a deployed web application accessible via a public URL on Streamlit Community Cloud. 
    The application is password-protected to prevent unauthorized access and uses a FAISS local file-based vector store 
    to implement a Retrieval Augmented Generation (RAG) pipeline. The application can answer questions based on a static 
    repository of PDF documents as well as PDFs uploaded by the user.

    Public officers often manage large volumes of documents, making the review process time-consuming and resource-intensive. 
    This model demonstrates how multiple publications on Singapore's urban development can be consolidated into a central 
    research database as a proof of concept. By leveraging large language models (LLM), the system streamlines document review and generates 
    coherent summaries, enabling faster and more efficient access to key insights.

    The project also includes documentation consisting of the following pages:
    """
)
st.markdown(
    """
    - **About Us:** Project scope, objectives, data source, and features.  
    - **Methodology:** Explanation of the technical implementation details and a flowchart illustrating the system workflow.
    """
)

# --- Objectives ---
st.header("Objectives")
st.write(
    """
    The objective of this project is to develop an application that enables users to upload PDF documents and 
    leverage an LLM to process and extract relevant information. The system will summarise 
    and present key insights for research or reference purposes, significantly reducing the time required to 
    review large volumes of lengthy documents.
    """
)

# --- Data Source ---
st.header("Data Source")
st.write(
    """
    At this stage, the system accepts only PDF documents, with a maximum file size of 200 MB. 
    The initial dataset comprises publications sourced from the 
    [Centre for Liveable Cities Knowledge Hub](https://knowledgehub.clc.gov.sg/), selected for its comprehensive 
    collection of books on Singapore’s urban development.
    """
)

# --- Features ---
st.header("Features")
st.markdown(
    """
    - **Secure Access:** Password login to prevent unauthorized use.  
    - **Preloaded Knowledge Base:** Static repository of documents serving as a base reference.  
    - **Document Upload:** Support for uploading PDF files (up to 200 MB) for analysis.  
    - **RAG Pipeline:** Multi-stage retrieval process (pre-retrieval, retrieval, post-retrieval) to maximise relevance and accuracy.  
    - **Prompt Engineering:** Designed to minimise prompt injection risks and prevent hallucinations.
    """
)
