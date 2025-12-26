import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
PDF_PATH = "retrieval-augmented-generation-options.pdf"
VECTOR_STORE_PATH = "faiss_index"

def ingest():
    """
    Ingests the PDF, splits it into chunks, and saves them to a local FAISS index.
    """
    print(f"--- XML Ingestion Started ---")
    
    # 1. Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found. Please ensure the file is in this directory.")
        return

    # 2. Load PDF
    print(f"Loading {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # 3. Split Text
    # Using RecursiveCharacterTextSplitter to respect sentence boundaries where possible
    # and keep chunks reasonably sized for retrieval.
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 4. Create Embeddings
    print("Initializing Embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5. Create Vector Store
    print("Creating FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 6. Save Locally
    print(f"Saving index to {VECTOR_STORE_PATH}...")
    vector_store.save_local(VECTOR_STORE_PATH)
    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    ingest()
