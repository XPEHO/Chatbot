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

# 1. Configuration & Environment
load_dotenv()
LLM_MODEL = os.getenv("MODEL_LLM")
EMB_MODEL = os.getenv("MODEL_EMBEDDING")
DOCS_DIR = 'exports'
PERSIST_DIR = 'vector-store'

# Initialize Models
embeddings = OpenAIEmbeddings(model=EMB_MODEL)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

# Initialize Vector Store
vector_store = Chroma(
    collection_name="collection_test",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

# 2. Document Loading Logic
def load_docs(folder):
    paths = list(Path(folder).rglob("*"))
    docs = []
    print(f"Loading documents from: {folder}")
    for p in tqdm(paths, desc="Load files"):
        try:
            if p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
            elif p.suffix.lower() == ".docx":
                docs.extend(Docx2txtLoader(str(p)).load())
            elif p.suffix.lower() in {".txt", ".md"}:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
        except Exception as e:
            print(f"Error loading {p.name}: {e}")
    return docs

# 3. LangGraph State & Nodes
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

prompt = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question}\n"
    "Context: {context}\n"
    "Answer:"
)

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# 4. Build the Graph
def build_graph():
    workflow = StateGraph(State)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    return workflow.compile()

# 5. Main Execution Flow
def main():
    # Only index if the directory exists and has files
    if os.path.exists(DOCS_DIR) and any(Path(DOCS_DIR).iterdir()):
        print("Starting document indexing...")
        raw_docs = load_docs(DOCS_DIR)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", ". ", " "]
        )
        all_splits = splitter.split_documents(raw_docs)
        if all_splits:
            vector_store.add_documents(documents=all_splits)
            print(f"Added {len(all_splits)} splits to vector store.")

    # Compile Graph
    app = build_graph()

    # Run Query
    query = query_entree
    # print(f"\nRunning query: {query}")
    
    with get_openai_callback() as cb:
        result = app.invoke({"question": query})
        print(f"--- Stats ---")
        # print(f"Tokens: {cb.total_tokens} | Cost: ${cb.total_cost:.6f}")
        print(f"--- Answer ---\n{result['answer']}")

if __name__ == "__main__":
    print("--- Script Starting ---")
    try:
        main()
    except Exception as e:
        print(f"--- CRITICAL ERROR ---\n{e}")
    print("--- Script Finished ---")
