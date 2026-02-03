import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from src.database import get_vector_db
from src.config import DATASET_PATH

def run_ingestion():
    """Loads a PDF, splits it into chunks, and uploads to Pinecone if empty."""
    
    if not os.path.exists(DATASET_PATH):
        print(f"--------------Error: Could not find PDF at {DATASET_PATH}--------------")
        return

    print(f"--------------Loading document: {DATASET_PATH}--------------")
    loader = PyPDFLoader(str(DATASET_PATH))
    documents = loader.load()

    print("--------------Splitting text into chunks--------------")
    # Using the specialized transformer splitter to ensure chunks match model limits
    splitter = SentenceTransformersTokenTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print(f"--------------Sending {len(chunks)} chunks to Pinecone--------------")
    # This calls our smart get_vector_db which handles the Pinecone logic
    db = get_vector_db(documents=chunks)
    
    print("--> Success. Ingestion complete! Your cloud database is ready.")

if __name__ == "__main__":
    run_ingestion()