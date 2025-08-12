# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# From helper_functions import check_password
from helper_functions.utility import check_password

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Import function to process documents (using the enhanced version with memory)
=======
# From helper functions import handle_document_upload
>>>>>>> parent of 2a32237 (update)
=======
# From helper functions import handle_document_upload
>>>>>>> parent of 2a32237 (update)
=======
# From helper functions import handle_document_upload
>>>>>>> parent of 2a32237 (update)
=======
# From helper functions import handle_document_upload
>>>>>>> parent of 2a32237 (update)
=======
# From helper functions import handle_document_upload
>>>>>>> parent of 2a32237 (update)
from logics.rag_pipeline import process_existing_documents, process_uploaded_document, answer_query_with_llm_filter

# Load environment variables from .env
load_dotenv()

# Check if the password is correct.  
if not check_password():  
    st.stop()

# Initialize session state for memory and functionality
if 'doc_uploaded' not in st.session_state:
    st.session_state['doc_uploaded'] = False

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'conversation_started' not in st.session_state:
    st.session_state['conversation_started'] = False

if 'existing_docs_loaded' not in st.session_state:
    st.session_state['existing_docs_loaded'] = False

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("Streamlit App")

# Process static PDFs on first load
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
if not st.session_state['existing_docs_loaded']:
    with st.spinner("Processing initial documents...this may take a few minutes"):
=======
if 'existing_docs_loaded' not in st.session_state:
    with st.spinner("Processing initial documents..."):
>>>>>>> parent of 2a32237 (update)
=======
if 'existing_docs_loaded' not in st.session_state:
    with st.spinner("Processing initial documents..."):
>>>>>>> parent of 2a32237 (update)
=======
if 'existing_docs_loaded' not in st.session_state:
    with st.spinner("Processing initial documents..."):
>>>>>>> parent of 2a32237 (update)
=======
if 'existing_docs_loaded' not in st.session_state:
    with st.spinner("Processing initial documents..."):
>>>>>>> parent of 2a32237 (update)
=======
if 'existing_docs_loaded' not in st.session_state:
    with st.spinner("Processing initial documents..."):
>>>>>>> parent of 2a32237 (update)
        try:
            process_existing_documents(folder_path="data")
            st.success("Static documents processed and vector store created.")
            st.session_state['existing_docs_loaded'] = True
        except Exception as e:
            st.error(f"Failed to process static documents: {e}")
            st.stop()

# Document Upload Section
st.subheader("📁 Document Management")
uploaded_file = st.file_uploader("Upload additional PDF", type=["pdf"])
if uploaded_file and st.button("Add to vector store"):
    with st.spinner("Processing uploaded document..."):
        try:
            process_uploaded_document(uploaded_file)
            st.success(f"✅ Uploaded document '{uploaded_file.name}' added to vector store.")
            st.session_state['doc_uploaded'] = True
        except Exception as e:
            st.error(f"❌ Failed to process uploaded document: {e}")

st.divider()

# Conversation History Section
if st.session_state['chat_history']:
    st.subheader("💬 Conversation History")
    
    # Create expandable section for history
    with st.expander(f"View Previous Q&A ({len(st.session_state['chat_history'])//2} exchanges)", expanded=False):
        for i in range(0, len(st.session_state['chat_history']), 2):
            if i + 1 < len(st.session_state['chat_history']):
                user_msg = st.session_state['chat_history'][i]
                assistant_msg = st.session_state['chat_history'][i + 1]
                
                st.write(f"**Q{(i//2)+1}:** {user_msg['content']}")
                st.write(f"**A{(i//2)+1}:** {assistant_msg['content']}")
                if i + 2 < len(st.session_state['chat_history']):  # Don't add separator after last item
                    st.write("---")
    
    # Add button to clear conversation
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🗑️ Clear History", type="secondary", help="Clear all conversation history"):
            st.session_state['chat_history'] = []
            st.session_state['conversation_started'] = False
            st.rerun()
    
    with col2:
        if st.button("📥 Download History", type="secondary", help="Download conversation as text"):
            # Create downloadable content
            history_text = "DocuMind Conversation History\n" + "="*50 + "\n\n"
            for i in range(0, len(st.session_state['chat_history']), 2):
                if i + 1 < len(st.session_state['chat_history']):
                    user_msg = st.session_state['chat_history'][i]
                    assistant_msg = st.session_state['chat_history'][i + 1]
                    history_text += f"Q{(i//2)+1}: {user_msg['content']}\n\n"
                    history_text += f"A{(i//2)+1}: {assistant_msg['content']}\n\n"
                    history_text += "-" * 50 + "\n\n"
            
            st.download_button(
                label="📄 Download as TXT",
                data=history_text,
                file_name="documind_conversation.txt",
                mime="text/plain"
            )

# Question Input Section
st.subheader("❓ Ask a Question" + (" (Follow-up questions will use conversation context)" if st.session_state['chat_history'] else ""))

# Sample questions
with st.expander("💡 Sample Questions", expanded=False):
    sample_questions = [
        "Tell me about Singapore's urban development",
        "Give me examples where HDB was involved in Singapore's urban development",
        "Which years were URA master plan mentioned?",
        "What are the key challenges in urban planning?",
        "How has Singapore's housing policy evolved over time?"
    ]
    for i, question in enumerate(sample_questions, 1):
        if st.button(f"{i}. {question}", key=f"sample_{i}"):
            st.session_state['sample_question'] = question

# Text area for question input
if 'sample_question' in st.session_state:
    user_prompt = st.text_area("Your question:", value=st.session_state['sample_question'], height=100)
    del st.session_state['sample_question']
else:
    user_prompt = st.text_area("Your question:", height=100)

# Action buttons
col1, col2 = st.columns([2, 1])

with col1:
    submit_button = st.button("🚀 Submit Question", type="primary", disabled=not user_prompt.strip())

with col2:
    new_conversation_button = st.button("🔄 New Conversation", type="secondary", help="Start a fresh conversation")

# Handle new conversation
if new_conversation_button:
    st.session_state['chat_history'] = []
    st.session_state['conversation_started'] = False
    st.success("✅ Started new conversation!")
    st.rerun()

# Handle question submission
if submit_button and user_prompt.strip():
    with st.spinner("🔍 Searching documents and generating answer..."):
        try:
            # Pass chat history to the function
            response = answer_query_with_llm_filter(
                user_prompt, 
                chat_history=st.session_state['chat_history']
            )
            
            # Display the response
            st.divider()
            st.subheader("💡 Answer:")
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
            
            # Limit history to prevent memory bloat (keep last 20 messages = 10 Q&A pairs)
            max_history_length = 20
            if len(st.session_state['chat_history']) > max_history_length:
                st.session_state['chat_history'] = st.session_state['chat_history'][-max_history_length:]
            
            # Success message
            st.success(f"✅ Answer generated! Conversation now has {len(st.session_state['chat_history'])//2} exchanges.")
            
        except Exception as e:
            st.error(f"❌ Error generating answer: {e}")
            # Optionally show more detailed error info in development
            with st.expander("Error Details (for debugging)"):
                st.code(str(e))

# Display current conversation context info
if st.session_state['chat_history']:
    st.info(f"💬 Conversation contains {len(st.session_state['chat_history'])//2} Q&A exchanges. Follow-up questions will reference previous context for better answers.")

# Footer with helpful information
st.divider()
with st.expander("ℹ️ How DocuMind Works"):
    st.markdown("""
    **DocuMind** uses advanced Retrieval-Augmented Generation (RAG) with conversation memory:
    
    1. **Document Processing**: PDFs are split into semantic chunks and stored in a vector database
    2. **Query Enhancement**: Follow-up questions are enhanced with conversation context
    3. **Smart Retrieval**: Uses Maximum Marginal Relevance (MMR) for diverse, relevant document retrieval
    4. **LLM Filtering**: Documents are scored for relevance before final answer generation
    5. **Memory**: Maintains conversation history for contextual follow-up questions
    
    **Tips for better results:**
    - Be specific in your questions
    - Use follow-up questions to drill down into topics
    - Check conversation history to see previous context
    - Upload additional PDFs to expand the knowledge base
    """)

# Add system status in sidebar
with st.sidebar:
    st.header("System Status")
    
    # Document status
    if st.session_state['existing_docs_loaded']:
        st.success("✅ Base documents loaded")
    else:
        st.warning("⏳ Loading base documents...")
    
    # Conversation status
    if st.session_state['chat_history']:
        st.info(f"💬 {len(st.session_state['chat_history'])//2} Q&A exchanges")
    else:
        st.info("💬 No conversation history")
    
    # Upload status
    if st.session_state['doc_uploaded']:
        st.success("📄 Additional documents uploaded")
    
    st.divider()
    st.markdown("**Memory Management:**")
    st.caption("• Keeps last 10 Q&A pairs")
    st.caption("• Context shared across questions")
    st.caption("• Use 'New Conversation' to reset")