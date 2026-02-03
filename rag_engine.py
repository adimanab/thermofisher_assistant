from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import GROQ_API_KEY
from vector_store import get_retriever
from systemPrompt import get_thermo_med_prompt

# Initialize components
prompt = get_thermo_med_prompt()
output_parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    retriever = get_retriever()
    
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
    return rag_chain

def rag_query(question: str) -> str:
    if not question.strip():
        return "**Please enter a question.**"
    
    chain = get_rag_chain()
    return chain.invoke(question)
