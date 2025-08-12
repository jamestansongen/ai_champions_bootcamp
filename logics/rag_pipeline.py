import os
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
import time
from typing import List, Dict, Any
=======
=======
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
from typing import List
>>>>>>> parent of 2a32237 (update)
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import time
from langchain.vectorstores import FAISS
from helper_functions.llm import get_completion_by_messages

from dotenv import load_dotenv

load_dotenv()

### Step 0. Setup ###

# Embedding model used
embedding_model = OpenAIEmbeddings(
    model='text-embedding-3-small',
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME"),  # uses gpt-4o-mini from your .env
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"), 
    messages=[
        {
            "role": "system",
            "content": "You are a researcher that answers based ONLY on the sources available to you. Examine what you have thoroughly and do not hallucinate."
        }
    ]
)
# Vector DB directory
vector_db_directory = "./vector_db"
# Collection name
collection_name='embedding_semantic'

# Function to check time
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Finished {func.__name__} in {end - start:.2f} seconds")
        return result
    return wrapper

### Step 1. load repository of documents ###

# Load pre-existing documents from folder
@timed
def load_documents_from_folder(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            try:
                loader = PyPDFLoader(full_path)
                data = loader.load()
                documents.extend(data)
                print(f"✅ Loaded {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
    return documents

# Allow uploaded files to be added as extra resources
@timed
def add_uploaded_documents(uploaded_file) -> List[Document]:
    from io import BytesIO
    import tempfile

    # Save uploaded file temporarily to disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_filepath = tmp_file.name
    
    try:
        loader = PyPDFLoader(temp_filepath)
        docs = loader.load()
        print(f"✅ Uploaded file loaded successfully")
    finally:
        os.remove(temp_filepath)

    return docs

### Step 2. Splitting documents into semantic chunks ###
@timed
def split_documents(documents: List[Document]) -> List[Document]:
    if not documents:
        raise ValueError("No documents to split.")

    # Step 1: Semantic chunking (preserves meaning)
    semantic_splitter = SemanticChunker(embedding_model)
    semantic_chunks = semantic_splitter.split_documents(documents)

    # Step 2: Recursive splitting (ensures size control & overlap)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200
    )

    final_chunks = []
    for chunk in semantic_chunks:
        if len(chunk.page_content) > 1000:
            smaller_chunks = recursive_splitter.split_documents([chunk])
            final_chunks.extend(smaller_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks

### Step 3. Storage ###

# Function to create or update FAISS vectorstore and persist
@timed  
def persist_vector_store(chunks: List[Document]):
    vectordb = FAISS.from_documents(
        chunks,
        embedding_model
    )
    vectordb.save_local("./faiss_index")  # Save index locally
    return vectordb

# Load persisted FAISS vectorstore
def load_vector_store():
    vectordb = FAISS.load_local(
        "./faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectordb


### Step 4. Retrieval - MMR ###
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
# MMR: When you want a balance of relevance and diversity, e.g., in exploratory or general-purpose retrieval.

>>>>>>> parent of 2a32237 (update)
=======
# MMR: When you want a balance of relevance and diversity, e.g., in exploratory or general-purpose retrieval.

>>>>>>> parent of 2a32237 (update)
=======
# MMR: When you want a balance of relevance and diversity, e.g., in exploratory or general-purpose retrieval.

>>>>>>> parent of 2a32237 (update)
=======
# MMR: When you want a balance of relevance and diversity, e.g., in exploratory or general-purpose retrieval.

>>>>>>> parent of 2a32237 (update)
=======
# MMR: When you want a balance of relevance and diversity, e.g., in exploratory or general-purpose retrieval.

>>>>>>> parent of 2a32237 (update)
=======
# MMR: When you want a balance of relevance and diversity, e.g., in exploratory or general-purpose retrieval.

>>>>>>> parent of 2a32237 (update)
def build_qa_chain(vectordb, strategy="mmr"):
    if strategy == "mmr":
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": 50}
        )
    else:
        raise ValueError(f"Unsupported retrieval strategy: {strategy}")

    return RetrievalQA.from_llm(llm=llm, retriever=retriever)

### Memory Management Functions ###

def format_chat_history(chat_history: List[Dict[str, str]], max_turns: int = 5) -> str:
    """Format chat history for context. Keep only last max_turns exchanges."""
    if not chat_history:
        return ""
    
    # Keep only the most recent exchanges
    recent_history = chat_history[-max_turns*2:] if len(chat_history) > max_turns*2 else chat_history
    
    formatted_history = []
    for entry in recent_history:
        if entry["role"] == "user":
            formatted_history.append(f"Previous Question: {entry['content']}")
        elif entry["role"] == "assistant":
            formatted_history.append(f"Previous Answer: {entry['content']}")
    
    return "\n".join(formatted_history)

def enhance_query_with_context(user_prompt: str, chat_history: List[Dict[str, str]]) -> str:
    """Enhance user query with conversation context to improve retrieval."""
    if not chat_history:
        return user_prompt
    
    # Create a context-enhanced query
    history_context = format_chat_history(chat_history, max_turns=3)
    
    context_prompt = f"""
Based on the following conversation history, rephrase or expand the current question to be more specific and standalone.

{history_context}

Current Question: {user_prompt}

Enhanced Question (be specific and include relevant context):"""
    
    try:
        enhanced_query = get_completion_by_messages([
            {"role": "system", "content": "You enhance questions to be more specific and standalone based on conversation context. Keep the enhanced question focused and relevant."},
            {"role": "user", "content": context_prompt}
        ])
        return enhanced_query.strip()
    except Exception as e:
        print(f"Error enhancing query: {e}")
        return user_prompt

### Step 5/6. Main functions with Memory Support ###

def answer_query_with_llm_filter(user_prompt: str, chat_history: List[Dict[str, str]] = None, strategy="mmr") -> str:
    """Main query function with conversation memory support."""
    if chat_history is None:
        chat_history = []
    
    vectordb = load_vector_store()
    
    # Step 1: Enhance query with conversation context
    enhanced_query = enhance_query_with_context(user_prompt, chat_history)
    print(f"Original query: {user_prompt}")
    print(f"Enhanced query: {enhanced_query}")
    
    # Step 2: Choose retriever based on strategy
    if strategy == "mmr":
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": 50}
        )
    else:
        raise ValueError(f"Unsupported retrieval strategy: {strategy}")
    
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    # Step 3: Retrieve raw documents using enhanced query
    raw_docs = retriever.get_relevant_documents(enhanced_query)
    
    # Step 4: Filter raw documents with LLM
=======
=======
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
    # Step 1: Retrieve raw docs
    raw_docs = retriever.get_relevant_documents(user_prompt)
    
    # Step 2: Filter docs with LLM
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
=======
>>>>>>> parent of 2a32237 (update)
    filtered_docs = filter_documents_with_llm(raw_docs, user_prompt)
    
    if not filtered_docs:
        return "Sorry, no relevant documents found to answer your question."
    
    # Step 5: Create context for final prompt with conversation history
    document_context = "\n\n".join(doc.page_content for doc in filtered_docs)
    conversation_context = format_chat_history(chat_history, max_turns=3)
    
    final_prompt = f"""
Answer the question using ONLY the context below. If the answer is not contained within the context, say "I don't know."

{"Conversation History:" if conversation_context else ""}
{conversation_context}

Document Context:
\"\"\" 
{document_context}
\"\"\"

Current Question: {user_prompt}
Answer:"""
    
    # Step 6: Generate answer with LLM
    response = get_completion_by_messages([
        {"role": "system", "content": "You are a researcher that answers based ONLY on the sources available to you. Examine what you have thoroughly and do not hallucinate. Consider the conversation history when answering follow-up questions."},
        {"role": "user", "content": final_prompt}
    ])
    
    return response


# Helper function to process documents (both preloaded and optional uploaded documents)
def process_all_documents(uploaded_file=None):
    # Load preloaded folder docs
    folder_docs = load_documents_from_folder("data")
    # Add uploaded docs if available
    if uploaded_file is not None:
        uploaded_docs = add_uploaded_documents(uploaded_file)
        all_docs = folder_docs + uploaded_docs
    else:
        all_docs = folder_docs
    # Split chunks
    chunks = split_documents(all_docs)
    # Persist vector store
    vectordb = persist_vector_store(chunks)
    return vectordb

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
@timed
def process_existing_documents(folder_path="data"):
    """Loads, splits, and persists vector store from static folder"""
    folder_docs = load_documents_from_folder(folder_path)
    chunks = split_documents(folder_docs)
    persist_vector_store(chunks)

@timed
def process_uploaded_document(uploaded_file):
    """Adds an uploaded file to the vector store incrementally"""
    uploaded_docs = add_uploaded_documents(uploaded_file)
    chunks = split_documents(uploaded_docs)

    vectordb = load_vector_store()
    vectordb.add_documents(chunks)
    persist_vector_store(chunks)  # Optionally re-save index

@timed
<<<<<<< HEAD
def filter_documents_with_llm(docs: List[Document], query: str, threshold: int = 3) -> List[Document]:
    """Helper function to filter relevant documents to use to answer question"""
=======
=======
@timed
def process_existing_documents(folder_path="data"):
    """Loads, splits, and persists vector store from static folder"""
    folder_docs = load_documents_from_folder(folder_path)
    chunks = split_documents(folder_docs)
    persist_vector_store(chunks)

@timed
def process_uploaded_document(uploaded_file):
    """Adds an uploaded file to the vector store incrementally"""
    uploaded_docs = add_uploaded_documents(uploaded_file)
    chunks = split_documents(uploaded_docs)

    vectordb = load_vector_store()
    vectordb.add_documents(chunks)
    persist_vector_store(chunks)  # Optionally re-save index

@timed
>>>>>>> parent of 2a32237 (update)
=======
@timed
def process_existing_documents(folder_path="data"):
    """Loads, splits, and persists vector store from static folder"""
    folder_docs = load_documents_from_folder(folder_path)
    chunks = split_documents(folder_docs)
    persist_vector_store(chunks)

@timed
def process_uploaded_document(uploaded_file):
    """Adds an uploaded file to the vector store incrementally"""
    uploaded_docs = add_uploaded_documents(uploaded_file)
    chunks = split_documents(uploaded_docs)

    vectordb = load_vector_store()
    vectordb.add_documents(chunks)
    persist_vector_store(chunks)  # Optionally re-save index

@timed
>>>>>>> parent of 2a32237 (update)
=======
@timed
def process_existing_documents(folder_path="data"):
    """Loads, splits, and persists vector store from static folder"""
    folder_docs = load_documents_from_folder(folder_path)
    chunks = split_documents(folder_docs)
    persist_vector_store(chunks)

@timed
def process_uploaded_document(uploaded_file):
    """Adds an uploaded file to the vector store incrementally"""
    uploaded_docs = add_uploaded_documents(uploaded_file)
    chunks = split_documents(uploaded_docs)

    vectordb = load_vector_store()
    vectordb.add_documents(chunks)
    persist_vector_store(chunks)  # Optionally re-save index

@timed
>>>>>>> parent of 2a32237 (update)
=======
@timed
def process_existing_documents(folder_path="data"):
    """Loads, splits, and persists vector store from static folder"""
    folder_docs = load_documents_from_folder(folder_path)
    chunks = split_documents(folder_docs)
    persist_vector_store(chunks)

@timed
def process_uploaded_document(uploaded_file):
    """Adds an uploaded file to the vector store incrementally"""
    uploaded_docs = add_uploaded_documents(uploaded_file)
    chunks = split_documents(uploaded_docs)

    vectordb = load_vector_store()
    vectordb.add_documents(chunks)
    persist_vector_store(chunks)  # Optionally re-save index

@timed
>>>>>>> parent of 2a32237 (update)
=======
@timed
def process_existing_documents(folder_path="data"):
    """Loads, splits, and persists vector store from static folder"""
    folder_docs = load_documents_from_folder(folder_path)
    chunks = split_documents(folder_docs)
    persist_vector_store(chunks)

@timed
def process_uploaded_document(uploaded_file):
    """Adds an uploaded file to the vector store incrementally"""
    uploaded_docs = add_uploaded_documents(uploaded_file)
    chunks = split_documents(uploaded_docs)

    vectordb = load_vector_store()
    vectordb.add_documents(chunks)
    persist_vector_store(chunks)  # Optionally re-save index

@timed
>>>>>>> parent of 2a32237 (update)
def filter_documents_with_llm(docs: List[Document], query: str, threshold: int = 5) -> List[Document]:
>>>>>>> parent of 2a32237 (update)
    filtered_docs = []
    for doc in docs:
        prompt = f"""
You are a researcher that answers based ONLY on the sources available to you.
Examine what you have thoroughly and do not hallucinate.

Your task is to determine whether this document is useful for answering the given question. Score this document from 0 to 10 for how useful it is to answer the question.

Question: {query}

Document content:
\"\"\"
{doc.page_content}
\"\"\"

Is this document useful for answering the question? Reply only with a score of 0 to 10."""
        
        response = get_completion_by_messages([
            {"role": "system", "content": "You are a researcher that answers based ONLY on the sources available to you. Examine what you have thoroughly and do not hallucinate."},
            {"role": "user", "content": prompt}
        ])
        
        try:
            score = int(response.strip())
            if score >= threshold:
                filtered_docs.append(doc)
        except ValueError:
            # If LLM gave something unexpected, we skip this doc
            continue

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    return filtered_docs

@timed
def process_existing_documents(folder_path="data"):
    """Helper function to process preloaded documents"""
    """Loads, splits, and persists vector store from static folder"""
    folder_docs = load_documents_from_folder(folder_path)
    chunks = split_documents(folder_docs)
    persist_vector_store(chunks)

@timed
def process_uploaded_document(uploaded_file):
    """Helper function to process optional uploaded documents"""
    """Adds an uploaded file to the vector store incrementally"""
    uploaded_docs = add_uploaded_documents(uploaded_file)
    chunks = split_documents(uploaded_docs)

    # Load existing vector store
    try:
        vectordb = load_vector_store()
        # Add new documents to existing vectorstore
        new_vectordb = FAISS.from_documents(chunks, embedding_model)
        vectordb.merge_from(new_vectordb)
        vectordb.save_local("./faiss_index")
    except Exception as e:
        # If no existing vector store, create new one
        print(f"Creating new vector store: {e}")
        persist_vector_store(chunks)
=======
    
    return filtered_docs
>>>>>>> parent of 2a32237 (update)
=======
    
    return filtered_docs
>>>>>>> parent of 2a32237 (update)
=======
    
    return filtered_docs
>>>>>>> parent of 2a32237 (update)
=======
    
    return filtered_docs
>>>>>>> parent of 2a32237 (update)
=======
    
    return filtered_docs
>>>>>>> parent of 2a32237 (update)
=======
    
    return filtered_docs
>>>>>>> parent of 2a32237 (update)
