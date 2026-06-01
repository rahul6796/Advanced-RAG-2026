from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from pathlib import Path

BASE_DIR = Path.cwd()

PDF_PATH = BASE_DIR / "documents" / "sustainable_development.pdf"

print(PDF_PATH)
print(PDF_PATH.exists())

# PDF_PATH = "../documents/sustainable_development.pdf"
CHROMA_DIR = "data/chroma_db"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5-mini"


def build_rag_chain():
    docs = PyPDFLoader(PDF_PATH).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150
    ).split_documents(docs)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_DIR
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "Use only the context below to answer the question. "
        "If the context does not contain the answer, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    
    chain = prompt | llm
    
    return chain, retriever
