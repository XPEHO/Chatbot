#!/usr/bin/env python3
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

load_dotenv()

BACKEND_MODEL_ID = os.getenv("BACKEND_MODEL_ID", "chatbot-rag")
LLM_MODEL = os.getenv("MODEL_LLM", "gpt-4o-mini")
EMB_MODEL = os.getenv("MODEL_EMBEDDING", "text-embedding-3-small")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector-store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "collection_test")
TOP_K = int(os.getenv("TOP_K", "4"))
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant for internal documentation. "
    "Use the provided context first. If context is missing, say it clearly.",
)


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = 0.2
    stream: bool | None = False
    max_tokens: int | None = None


app = FastAPI(title="Chatbot RAG API", version="0.1.0")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join([p for p in parts if p])
    return str(content)


def _get_vector_store() -> Chroma | None:
    try:
        embeddings = OpenAIEmbeddings(model=EMB_MODEL)
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_STORE_DIR,
        )
    except Exception:
        return None


def _to_source_ref(metadata: dict[str, Any]) -> str:
    raw_source = str(metadata.get("source", "unknown"))
    source_name = Path(raw_source).name if raw_source else "unknown"
    page = metadata.get("page")
    if page is None:
        return source_name
    try:
        page_num = int(page) + 1
    except Exception:
        return f"{source_name} (page {page})"
    return f"{source_name} (page {page_num})"


def _retrieve_context(question: str) -> tuple[str, list[str]]:
    vector_store = _get_vector_store()
    if vector_store is None:
        return "", []
    try:
        docs = vector_store.similarity_search(question, k=TOP_K)
    except Exception:
        return "", []
    chunks: list[str] = []
    refs: list[str] = []
    for doc in docs:
        ref = _to_source_ref(doc.metadata)
        if ref not in refs:
            refs.append(ref)
        chunks.append(f"[{ref}]\n{doc.page_content}")
    return "\n\n".join(chunks), refs


def _build_prompt(messages: list[ChatMessage]) -> tuple[str, list[str]]:
    user_messages = [m for m in messages if m.role == "user"]
    question = _content_to_text(user_messages[-1].content).strip() if user_messages else ""
    if not question:
        raise HTTPException(status_code=400, detail="No user message provided")

    history_lines: list[str] = []
    for msg in messages[-8:]:
        if msg.role not in {"user", "assistant"}:
            continue
        text = _content_to_text(msg.content).strip()
        if text:
            history_lines.append(f"{msg.role}: {text}")

    context, refs = _retrieve_context(question)
    context_block = context if context else "No relevant context found in vector store."
    history_block = "\n".join(history_lines)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Conversation:\n{history_block}\n\n"
        f"User question:\n{question}\n\n"
        "Answer in the same language as the question. Be concise and factual. "
        "If context is used, cite in-text references like [source]."
    )
    return prompt, refs


def _generate_answer(request: ChatCompletionRequest) -> tuple[str, int]:
    prompt, refs = _build_prompt(request.messages)
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=request.temperature if request.temperature is not None else 0.2,
    )
    result = llm.invoke(prompt)
    text = result.content if isinstance(result.content, str) else str(result.content)
    sources_block = "Sources used:\n"
    if refs:
        sources_block += "\n".join(f"- {ref}" for ref in refs)
    else:
        sources_block += "- none (no document chunk retrieved)"
    text = f"{text.strip()}\n\n{sources_block}"
    return text, len(prompt)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": BACKEND_MODEL_ID,
                "object": "model",
                "owned_by": "chatbot",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    try:
        answer, prompt_chars = _generate_answer(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model_id = request.model or BACKEND_MODEL_ID

    if request.stream:
        def event_stream():
            chunk_1 = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk_1)}\n\n"

            chunk_2 = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk_2)}\n\n"

            chunk_3 = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk_3)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_chars // 4,
            "completion_tokens": len(answer) // 4,
            "total_tokens": (prompt_chars + len(answer)) // 4,
        },
    }
