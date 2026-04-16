import os
from typing import List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

load_dotenv()
LLM_MODEL = os.getenv("MODEL_LLM")
EMB_MODEL = os.getenv("MODEL_EMBEDDING")
PERSIST_DIR = os.getenv("VECTOR_STORE_DIR", "vector-store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "collection_test")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class RAGCore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMB_MODEL)
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
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
            docs = self.vector_store.similarity_search(state["question"], k=5)
            return {"context": docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            sources = list(set([os.path.basename(doc.metadata.get("source", "Unknown")) for doc in state["context"]]))
            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = self.llm.invoke(messages)
            source_text = "\n\n**Sources:**\n- " + "\n- ".join(sources)
            return {"answer": response.content + source_text}

        workflow = StateGraph(State)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        return workflow.compile()

    def query(self, text: str) -> str:
        result = self.graph.invoke({"question": text})
        return result["answer"]
