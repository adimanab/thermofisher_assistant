import os
import gradio as gr

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import (
    SentenceTransformersTokenTextSplitter
)
from langchain_chroma import Chroma

from systemPrompt import get_thermo_med_prompt


# ==============================
# CONFIG
# ==============================
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

DATASET_PATH = "dataset.pdf"
PERSIST_DIR = "pharma_db"

os.makedirs(PERSIST_DIR, exist_ok=True)


# ==============================
# EMBEDDINGS
# ==============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==============================
# VECTOR DATABASE
# ==============================
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)


# ==============================
# LOAD & INDEX PDF
# ==============================
if os.path.exists(DATASET_PATH):

    if len(db.get()["ids"]) == 0:
        print("📄 Indexing PDF...")

        loader = PyPDFLoader(DATASET_PATH)
        documents = loader.load()

        splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)
        db.add_documents(chunks)

        print("✅ PDF indexed successfully.")

else:
    print("⚠️ dataset.pdf not found.")


# ==============================
# PROMPT & PARSER
# ==============================
prompt = get_thermo_med_prompt()
output_parser = StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ==============================
# RAG QUERY FUNCTION
# ==============================
def run_query(question: str) -> str:

    if not question.strip():
        return "Please enter a question."

    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | output_parser
    )

    return rag_chain.invoke(question)


# ==============================
# GRADIO UI
# ==============================
interface = gr.Interface(
    fn=run_query,
    inputs=gr.Textbox(
        label="Question",
        placeholder="Ask a cancer-related question..."
    ),
    outputs=gr.Textbox(
        label="Response",
        lines=10
    ),
    title="Thermo Med Assistant",
    description="Ask evidence-based questions related to cancer."
)

interface.launch(server_name="0.0.0.0", server_port=7860)