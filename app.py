import os
import gradio as gr
import base64

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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "thermo_logo.jpg")

title_html = f"""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
    <img src="/file={LOGO_PATH}" style="height: 50px; width: auto; object-fit: contain;" />
    <h1 style="margin: 0; font-size: 24px; font-weight: 600;">Thermo Med Assistant</h1>
</div>
"""



with gr.Blocks(
    css="""
    #response_box {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 14px;
        min-height: 260px;
        background-color: white;
        overflow-y: auto;
        font-size: 14px;
        line-height: 1.6;
    }
    """
) as interface:

    # ---------------------------------
    # HEADER (LOGO + TITLE)
    # ---------------------------------
    gr.HTML(title_html)

    gr.Markdown("Ask genetic-based questions related to ThermoFisher Scientific.")

    # ---------------------------------
    # RESPONSE AREA
    # ---------------------------------
    response_output = gr.Markdown(
        label="Response",
        elem_id="response_box"
    )

    # ---------------------------------
    # INPUT AREA
    # ---------------------------------
    question_input = gr.Textbox(
        label="Your question",
        placeholder="Ask any query related to ThermoFisher Scientific...",
        lines=2
    )

    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear")

    submit_btn.click(run_query, question_input, response_output)
    clear_btn.click(lambda: "", None, response_output)


# ---------------------------------
# LAUNCH
# ---------------------------------
interface.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[BASE_DIR])