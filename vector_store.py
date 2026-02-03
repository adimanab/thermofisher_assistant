import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_chroma import Chroma
from config import DATASET_PATH, PERSIST_DIR

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize/Load Vector DB
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

def initialize_vector_db():
    """Indexes the PDF dataset if the database is empty."""
    if os.path.exists(DATASET_PATH) and len(db.get()["ids"]) == 0:
        print(f"📄 Indexing {DATASET_PATH}...")

        loader = PyPDFLoader(DATASET_PATH)
        documents = loader.load()

        splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)
        db.add_documents(chunks)

        print("✅ Indexing completed")

def get_retriever(k=5):
    return db.as_retriever(search_kwargs={"k": k})
