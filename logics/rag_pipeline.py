import os
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
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
            "content": "You are an assistant that answers based ONLY on provided context. Do not hallucinate."
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

### Step 2. Splitting documents into sematic chunks ###
@timed
def split_documents(documents: List[Document]) -> List[Document]:
    if not documents:
        raise ValueError("No documents to split.")
    
    text_splitter = SemanticChunker(embedding_model)
    chunks = text_splitter.split_documents(documents)
    return chunks


### Step 3. Storage ###

# Function to create or update Chroma vectorstore and persist
@timed  
def persist_vector_store(chunks: List[Document]):
    vectordb = FAISS.from_documents(
        chunks,
        embedding_model
    )
    vectordb.save_local("./faiss_index")  # Save index locally
    return vectordb

# Load persisted Chroma vectorstore
def load_vector_store():
    vectordb = FAISS.load_local(
        "./faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectordb


### Step 4. Retrieval - MMR ###
# MMR: When you want a balance of relevance and diversity, e.g., in exploratory or general-purpose retrieval.

def build_qa_chain(vectordb, strategy="mmr"):
    if strategy == "mmr":
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10}
        )
    else:
        raise ValueError(f"Unsupported retrieval strategy: {strategy}")

    return RetrievalQA.from_llm(llm=llm, retriever=retriever)

### Step 5/6. Main functions (post-retrieval implemented in answer_query_with_llm_filter)

# Main query function for Streamlit app
def answer_query_with_llm_filter(user_prompt: str, strategy="mmr") -> str:
    vectordb = load_vector_store()
    
    # Choose retriever based on strategy
    if strategy == "mmr":
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8}
        )
    else:
        raise ValueError(f"Unsupported retrieval strategy: {strategy}")
    
    # Step 1: Retrieve raw docs
    raw_docs = retriever.get_relevant_documents(user_prompt)
    
    # Step 2: Filter docs with LLM
    filtered_docs = filter_documents_with_llm(raw_docs, user_prompt)
    
    if not filtered_docs:
        return "Sorry, no relevant documents found to answer your question."
    
    # Step 3: Create context for final prompt
    context = "\n\n".join(doc.page_content for doc in filtered_docs)
    
    final_prompt = f"""
Answer the question using ONLY the context below. If the answer is not contained within the context, say "I don't know."

Context:
\"\"\" 
{context}
\"\"\"

Question: {user_prompt}
Answer:"""
    
    # Step 4: Generate answer with LLM
    response = get_completion_by_messages([
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
def filter_documents_with_llm(docs: List[Document], query: str) -> List[Document]:
    filtered_docs = []
    for doc in docs:
        prompt = f"""
You are an AI assistant helping to filter documents for a question.

Question: {query}

Document content:
\"\"\"
{doc.page_content}
\"\"\"

Is this document useful for answering the question? Reply only with 'Yes' or 'No'."""
        
        response = get_completion_by_messages([
            {"role": "user", "content": prompt}
        ])
        
        if "yes" in response.lower():
            filtered_docs.append(doc)
    
    return filtered_docs