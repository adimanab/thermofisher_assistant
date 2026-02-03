import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from src.config import DATASET_PATH, PINECONE_API_KEY, PINECONE_INDEX_NAME

def get_vector_db(documents=None):
    # 1. Initialize Embeddings (HuggingFace runs locally)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Initialize Pinecone Client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 3. Create index if it doesn't exist
    if not pc.has_index(PINECONE_INDEX_NAME):
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384, # Matches MiniLM-L6-v2 output
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # 4. Connect to the Index
    index = pc.Index(PINECONE_INDEX_NAME)   #connection object with methods like upsert etc
    stats = index.describe_index_stats()    #dictionary of metadata of the data, if empty then run pdf loader

    # 5. SMART LOGIC: Upload only if index is empty
    if stats['total_vector_count'] == 0:
        if os.path.exists(DATASET_PATH):
            print("📤 Pinecone index is empty. Processing PDF and uploading...")
            loader = PyPDFLoader(str(DATASET_PATH))
            splitter = SentenceTransformersTokenTextSplitter(chunk_size=500, chunk_overlap=50)
            documents = splitter.split_documents(loader.load())
            
            # This creates embeddings locally and sends them to Pinecone
            db = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME
            )
            return db
        else:
            raise FileNotFoundError(f"Index is empty and {DATASET_PATH} not found!")
    
    # 6. If not empty, just connect and return
    print(f"✅ Connected to Pinecone. Using {stats['total_vector_count']} existing vectors.")
    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )