import os
import base64
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


# =====================================================
# CONFIG
# =====================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

DATASET_PATH = "dataset.pdf"
PERSIST_DIR = "pharma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)


# =====================================================
# EMBEDDINGS & VECTOR DB
# =====================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)


# =====================================================
# LOAD & INDEX PDF (ONCE)
# =====================================================
if os.path.exists(DATASET_PATH) and len(db.get()["ids"]) == 0:
    print("📄 Indexing dataset.pdf...")

    loader = PyPDFLoader(DATASET_PATH)
    documents = loader.load()

    splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    db.add_documents(chunks)

    print("✅ Indexing completed")


# =====================================================
# PROMPT & OUTPUT PARSER
# =====================================================
prompt = get_thermo_med_prompt()
output_parser = StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =====================================================
# RAG QUERY FUNCTION (DO NOT OVERRIDE)
# =====================================================
def rag_query(question: str) -> str:

    if not question.strip():
        return "**Please enter a question.**"

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


# =====================================================
# LOGO (BASE64 — BROWSER SAFE)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "thermo_logo.jpg")

def encode_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")

logo_b64 = encode_image(LOGO_PATH)

header_html = f"""
<div style="
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 0;
">
    <img
        src="data:image/jpeg;base64,{logo_b64}"
        style="height: 50px; object-fit: contain;"
    />
    <span style="font-size: 22px; font-weight: 600;">
        Thermo Med Assistant
    </span>
</div>
"""


# =====================================================
# GRADIO UI
# =====================================================
with gr.Blocks(
    css="""
    #response_box {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 14px;
        min-height: 260px;
        background-color: white;
        font-size: 14px;
        line-height: 1.6;
    }
    """
) as interface:

    # HEADER
    gr.HTML(header_html)

    gr.Markdown(
        "Ask **genetic-based questions** related to **ThermoFisher Scientific**."
    )

    response_output = gr.Markdown(elem_id="response_box")

    question_input = gr.Textbox(
        placeholder="Ask any query related to ThermoFisher Scientific...",
        lines=2
    )

    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear")

    submit_btn.click(
        fn=rag_query,
        inputs=question_input,
        outputs=response_output
    )

    clear_btn.click(
        fn=lambda: "",
        inputs=None,
        outputs=response_output
    )


# =====================================================
# LAUNCH
# =====================================================
interface.launch(
    server_name="0.0.0.0",
    server_port=7860
)
