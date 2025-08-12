import streamlit as st

# from helper_functions import utility check_password
from helper_functions.utility import check_password

# Check if the password is correct.  
if not check_password():  
    st.stop()

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("About this App")

st.write("This is a Streamlit App that demonstrates how to use the OpenAI API to generate text completions.")

<<<<<<< HEAD
# Display flowchart if exists
if os.path.exists(png_path):
    st.image(png_path, caption="DocuMind flowchart", use_column_width=True)
else:
    st.warning(f"Flowchart not found at: {png_path}")
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

# Methodology description
st.markdown("""
At the start of the Streamlit interface, users are prompted to enter a password.  
If the password is incorrect, the system stops whereas if it is correct, it proceeds to load preloaded static documents stored as PDF files using a Retrieval-Augmented Generation (RAG) pipeline.  
Users can also upload additional PDF documents, and the same RAG pipeline is applied to them.  

The RAG process consists of three stages.  
1. Pre-retrieval stage: Documents are semantically segmented using a `SemanticChunker` to preserve meaning and coherence.  
These chunks are then recursively split with a `RecursiveCharacterTextSplitter` (chunk_size = 1000, chunk_overlap = 200) to ensure manageable size and contextual continuity.  
2. Retrieval stage: The system applies Maximum Marginal Relevance (MMR) to select the top 50 candidates from the vector store, then filters down to 15 documents while balancing relevance and diversity, along with reducing redundancy (fetch_k = 50, k = 15).  
3. Post-retrieval stage: An LLM-based scoring filter evaluates each retrieved document for usefulness on a 0–10 scale, discarding any with a score below 5 to improve answer quality.  

Prompt engineering safeguards are embedded throughout by instructing the LLM to act as a researcher and limiting responses only to the provided documents. The system instructions are repeated before and after the prompt to reduce prompt injection risks, and hallucinations are discouraged.  

Once this process is complete, the user can pose a relevant question, to which the system will respond with an answer if possible. If it is unable to provide a response, it will return with "I don’t know" and await the next interaction.
""")
=======
>>>>>>> parent of 1429dc3 (final update i hope)
=======
>>>>>>> parent of 1429dc3 (final update i hope)
=======
>>>>>>> parent of 1429dc3 (final update i hope)
=======
with st.expander("How to use this App"):
    st.write("1. Enter your prompt in the text area.")
    st.write("2. Click the 'Submit' button.")
    st.write("3. The app will generate a text completion based on your prompt.")

# Execute the project and add flowchart
>>>>>>> parent of 2a32237 (update)
