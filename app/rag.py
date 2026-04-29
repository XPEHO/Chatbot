import os
import logging
from typing import List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
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
        self.base_retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K * 3})
        self.compressor = FlashrankRerank(top_n=TOP_K)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.base_retriever,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        prompt = ChatPromptTemplate.from_template(
            "Tu es un assistant pour répondre à des questions "
            "Utilise le contexte ci-dessous pour répondre à la question"
            "Si le context fourni ne contient pas assez d'informations pour répondre, dis le clairement plutôt que d'inventer\n\n"
            "Contexte: {context}\n\n"
            "Question: {question}\n\n"
            "Réponse:"
        )

        def retrieve(state: State):
            question = state["question"]

            raw_docs = self.base_retriever.invoke(question)
            for i, doc in enumerate(raw_docs):
                source = os.path.basename(doc.metadata.get("source", "?"))

            reranked_docs = self.retriever.invoke(question)
            for i, doc in enumerate(reranked_docs):
                source = os.path.basename(doc.metadata.get("source", "?"))

            return {"context": reranked_docs}

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
