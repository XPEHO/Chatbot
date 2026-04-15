import os
from pathlib import Path
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv
from tqdm import tqdm

# LangChain & LangGraph Imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain_community.callbacks.manager import get_openai_callback

# Configuration & Environment
load_dotenv()
# LLM_MODEL = os.getenv("MODEL_LLM")
EMB_MODEL = os.getenv("MODEL_EMBEDDING", "text-embedding-3-small")
DOCS_DIR = 'exports'
PERSIST_DIR = 'vector-store'


import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
EMB_MODEL = os.getenv("MODEL_EMBEDDING", "text-embedding-3-small")
DOCS_DIR = 'exports'
PERSIST_DIR = 'vector-store'

def ingest():
    if not os.path.exists(DOCS_DIR):
        print(f"Error: {DOCS_DIR} folder not found.")
        return
    
    # Document Loading Logic
    print("--- Loading Documents ---")
    paths = list(Path(DOCS_DIR).rglob("*"))
    raw_docs = []
    
    for p in paths:
        try:
            if p.suffix.lower() == ".pdf":
                loader = PyMuPDFLoader(str(p))
                raw_docs.extend(loader.load())
            elif p.suffix.lower() == ".docx":
                raw_docs.extend(Docx2txtLoader(str(p)).load())
            elif p.suffix.lower() in {".txt", ".md"}:
                raw_docs.extend(TextLoader(str(p), encoding="utf-8").load())
        except Exception as e:
            print(f"Skipping {p.name} due to error: {e}")

    if not raw_docs:
        print("No documents found to index.")
        return

    print(f"Splitting {len(raw_docs)} documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    all_splits = splitter.split_documents(raw_docs)

    print(f"Generating embeddings and saving to {PERSIST_DIR}...")
    embeddings = OpenAIEmbeddings(model=EMB_MODEL)
    
    # Create the vector store on disk
    Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="collection_test"
    )
    print("Indexing complete!")

if __name__ == "__main__":
    ingest()

