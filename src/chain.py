# src/chain.py
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.database import get_vector_db
from src.prompts import get_thermo_med_prompt
from src.config import GROQ_API_KEY

def format_docs(docs):
    """Combines retrieved document chunks into a single string for context."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    """
    Creates and returns the RAG (Retrieval-Augmented Generation) chain.
    This chain connects the Pinecone vector database to the Groq LLM.
    """
    # 1. Connect to our cloud database (from_existing_index logic)
    db = get_vector_db() 
    
    # 2. Setup the Retriever (looks for top 5 most relevant chunks)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # 3. Initialize the Groq LLM (Llama 3.1)
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        api_key=GROQ_API_KEY, 
        temperature=0
    )
    
    # 4. Get the system instructions for ThermoFisher domain
    prompt = get_thermo_med_prompt()
    
    # 5. Define the LCEL (LangChain Expression Language) Chain
    # This pipeline flows from: Input -> Retrieval -> Formatting -> Prompt -> LLM -> String
    chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough()
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain