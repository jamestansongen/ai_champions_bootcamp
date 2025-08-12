import streamlit as st
import os

# Set page configuration
st.set_page_config(
    page_title="Methodology",
    layout="centered"
)

# Page title
st.title("Methodology")

# Path to the PNG file
png_path = os.path.join(os.path.dirname(__file__), "..", "flowchart.png")

# Display flowchart if exists
if os.path.exists(png_path):
    st.image(png_path, caption="DocuMind flowchart", use_container_width=True)
else:
    st.warning(f"Flowchart not found at: {png_path}")

# Methodology description
st.markdown("""
At the start of the Streamlit interface, users are prompted to enter a password.  
If the password is incorrect, the system stops whereas if it is correct, it proceeds to load  
preloaded static documents stored as PDF files using a Retrieval-Augmented Generation (RAG) pipeline.  
Users can also upload additional PDF documents, and the same RAG pipeline is applied to them.  

The RAG process consists of three stages.  
1. Pre-retrieval stage: Documents are semantically segmented using a `SemanticChunker` to preserve meaning and coherence.  
These chunks are then recursively split with a `RecursiveCharacterTextSplitter` (chunk_size = 1000, chunk_overlap = 200)  
to ensure manageable size and contextual continuity.  
2. Retrieval stage: The system applies Maximum Marginal Relevance (MMR) to select the top 50 candidates from the vector store,  
then filters down to 15 documents while balancing relevance and diversity, along with reducing redundancy (fetch_k = 50, k = 15).  
3. Post-retrieval stage: An LLM-based scoring filter evaluates each retrieved document for usefulness on a 0–10 scale,  
discarding any with a score below 5 to improve answer quality.  

Prompt engineering safeguards are embedded throughout by instructing the LLM to act as a researcher  
and limiting responses only to the provided documents. The system instructions are repeated before and after the prompt  
to reduce prompt injection risks, and hallucinations are discouraged.  

Once this process is complete, the user can pose a relevant question,  
to which the system will respond with an answer if possible.  
If it is unable to provide a response, it will return with "I don’t know" and await the next interaction.
""")