# Import relevant libraries
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Import function to check password
from helper_functions.utility import check_password

# Import function to process documents (using the enhanced version with memory)
from logics.rag_pipeline import process_existing_documents, process_uploaded_document, answer_query_with_llm_filter

# Load environment variables from .env
load_dotenv()

# Check if the password is correct.  
if not check_password():  
    st.stop()

# Initialize session state for memory
if 'doc_uploaded' not in st.session_state:
    st.session_state['doc_uploaded'] = False

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'conversation_started' not in st.session_state:
    st.session_state['conversation_started'] = False

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

# Display conversation history if it exists
if st.session_state['chat_history']:
    st.subheader("Conversation History")
    
    # Create expandable section for history
    with st.expander("View Previous Q&A", expanded=False):
        for i in range(0, len(st.session_state['chat_history']), 2):
            if i + 1 < len(st.session_state['chat_history']):
                user_msg = st.session_state['chat_history'][i]
                assistant_msg = st.session_state['chat_history'][i + 1]
                
                st.write(f"**Q{(i//2)+1}:** {user_msg['content']}")
                st.write(f"**A{(i//2)+1}:** {assistant_msg['content']}")
                st.write("---")
    
    # Add button to clear conversation
    if st.button("🗑️ Clear Conversation", type="secondary"):
        st.session_state['chat_history'] = []
        st.session_state['conversation_started'] = False
        st.rerun()

st.subheader("Ask a question" + (" (Follow-up questions will use conversation context)" if st.session_state['chat_history'] else ""))

user_prompt = st.text_area("Your question (e.g. Tell me about Singapore's urban development/Give me examples where HDB was involved in Singapore's urban development/Which years were URA master plan mentioned?)", height=150)

col1, col2 = st.columns([1, 1])

with col1:
    submit_button = st.button("Submit Question", type="primary")

with col2:
    new_conversation_button = st.button("Start New Conversation", type="secondary")

# Handle new conversation
if new_conversation_button:
    st.session_state['chat_history'] = []
    st.session_state['conversation_started'] = False
    st.rerun()

# Handle question submission
if submit_button and user_prompt.strip():
    with st.spinner("Generating answer..."):
        try:
            # Pass chat history to the function
            response = answer_query_with_llm_filter(
                user_prompt, 
                chat_history=st.session_state['chat_history']
            )
            
            # Display the response
            st.subheader("Answer:")
            st.write(response)
            
            # Update chat history
            st.session_state['chat_history'].append({
                "role": "user",
                "content": user_prompt
            })
            st.session_state['chat_history'].append({
                "role": "assistant", 
                "content": response
            })
            st.session_state['conversation_started'] = True
            
            # Optional: Limit history to prevent memory bloat
            max_history_length = 20  # Keep last 10 Q&A pairs
            if len(st.session_state['chat_history']) > max_history_length:
                st.session_state['chat_history'] = st.session_state['chat_history'][-max_history_length:]
            
        except Exception as e:
            st.error(f"Error generating answer: {e}")

# Display current conversation context info
if st.session_state['chat_history']:
    st.info(f"💬 Conversation contains {len(st.session_state['chat_history'])//2} Q&A exchanges. Follow-up questions will reference previous context.")