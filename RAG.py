import os
import time
import uuid
from pathlib import Path
from typing import List, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

load_dotenv()
LLM_MODEL = os.getenv("MODEL_LLM")
EMB_MODEL = os.getenv("MODEL_EMBEDDING")
DOCS_DIR = 'exports'
PERSIST_DIR = 'vector-store'

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAGCore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMB_MODEL)
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
        self.vector_store = Chroma(
            collection_name="collection_test",
            embedding_function=self.embeddings,
            persist_directory=PERSIST_DIR,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        prompt = ChatPromptTemplate.from_template(
            "You are an assistant for question-answering tasks. Use the following context to answer.\n"
            "Context: {context}\nQuestion: {question}\nAnswer:"
        )

        def retrieve(state: State):
            # Optimized search: fetch top 5 relevant chunks
            docs = self.vector_store.similarity_search(state["question"], k=5)
            return {"context": docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            sources = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in state["context"]]))
            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = self.llm.invoke(messages)
            source_text = "\n\n**Sources:**\n- " + "\n- ".join(sources)
            final_answer = response.content + source_text
            return {"answer": final_answer}

        workflow = StateGraph(State)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        return workflow.compile()

    def query(self, text: str):
        return self.graph.invoke({"question": text})


app = FastAPI(title="AnythingLLM Custom Bridge")
engine = RAGCore()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "custom-rag"

@app.post("/v1/chat/completions")
async def chat_endpoint(request: ChatCompletionRequest):
    user_msg = request.messages[-1].content
    result = engine.query(user_msg)
    
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "message": {"role": "assistant", "content": result["answer"]},
            "finish_reason": "stop",
            "index": 0
        }]
    }

def run_ingestion():
    if not os.path.exists(DOCS_DIR): return
    
    paths = list(Path(DOCS_DIR).rglob("*"))
    new_docs = []
    for p in tqdm(paths, desc="Processing files"):
        if p.suffix.lower() == ".pdf": new_docs.extend(PyMuPDFLoader(str(p)).load())
        elif p.suffix.lower() == ".docx": new_docs.extend(Docx2txtLoader(str(p)).load())
        elif p.suffix.lower() in {".txt", ".md"}: new_docs.extend(TextLoader(str(p)).load())
    
    if new_docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = splitter.split_documents(new_docs)
        engine.vector_store.add_documents(splits)
        print(f"Successfully indexed {len(splits)} chunks.")

if __name__ == "__main__":
    import uvicorn
    run_ingestion()
    uvicorn.run(app, host="0.0.0.0", port=8000)