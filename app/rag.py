import os
from typing import List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, StateGraph

load_dotenv()
LLM_MODEL = os.getenv("MODEL_LLM")
EMB_MODEL = os.getenv("MODEL_EMBEDDING")
PERSIST_DIR = os.getenv("VECTOR_STORE_DIR", "vector-store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "collection_test")
TOP_K = int(os.getenv("TOP_K", "4"))


class State(TypedDict):
    question: str
    history: List[dict]
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
            "You are an assistant for question-answering tasks. "
            "Use ONLY the context below to answer. "
            "If the context does not contain enough information to answer, say so clearly instead of guessing.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        def retrieve(state: State):
            results = self.vector_store.similarity_search_with_relevance_scores(
                state["question"], k=TOP_K * 3
            )
            docs = [doc for doc, score in results if score >= 0.5][:TOP_K]
            return {"context": docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            sources = list(set([os.path.basename(doc.metadata.get("source", "Unknown")) for doc in state["context"]]))
            system = prompt.invoke({"question": state["question"], "context": docs_content})
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in state.get("history", [])
            ]
            response = self.llm.invoke(history + system.to_messages())
            source_text = "\n\n**Sources:**\n- " + "\n- ".join(sources) if sources else ""
            return {"answer": response.content + source_text}

        workflow = StateGraph(State)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        return workflow.compile()

    def query(self, text: str, history: List[dict] = []) -> str:
        result = self.graph.invoke({"question": text, "history": history})
        return result["answer"]
