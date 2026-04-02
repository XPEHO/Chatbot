#!/usr/bin/env python3
import json
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

BACKEND_MODEL_ID = os.getenv("BACKEND_MODEL_ID", "chatbot-rag")


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    stream: bool | None = False
    max_tokens: int | None = None


app = FastAPI(title="Chatbot API", version="0.1.0")


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

    answer = f"Reponse bidon API."
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model_id = request.model or BACKEND_MODEL_ID

    if request.stream:
        ##réponse compatible ANYTHINGLLM
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
    }
