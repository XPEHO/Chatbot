# Chatbot

This project provides:
- BookStack page export to PDF
- Vector store indexing (Chroma)
- A RAG chatbot API (OpenAI-compatible endpoints)
- An Open WebUI interface connected to the RAG API

## Prerequisites

- Python 3.11+
- Docker + Docker Compose
- OpenAI API key with active billing/quota
- BookStack API token (for export only)

## Environment setup

Create `.env` from `.env.example` and fill values:

```env
OPENAI_API_KEY=sk-proj-your-api-key
MODEL_LLM=yout_model
MODEL_EMBEDDING=your_embedding_model

BOOKSTACK_URL=your_url
BOOKSTACK_TOKEN_ID=your_bookstack_token_id
BOOKSTACK_TOKEN_SECRET=your_bookstack_token_secret
```

## Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Export BookStack pages as PDFs

```bash
python export_pages.py
```

PDF files are written to `exports/`.

## 2) Build / refresh vector store

```bash
python reload_vector_store.py
```

This indexes documents from `exports/` into `vector-store/`.

## 3) Run chatbot API + Open WebUI

```bash
docker compose up --build
```

Access:
- Open WebUI: `http://localhost:3000`
- Chatbot API health: `http://localhost:8000/health`

Open WebUI is configured to call the local chatbot backend through OpenAI-compatible routes:
- `GET /v1/models`
- `POST /v1/chat/completions`

## 4) Expose AnythingLLM on internet (secure minimal setup)

Files:
- `docker-compose.anythingllm.secure.yml`
- `deploy/Caddyfile`

This stack puts Caddy in front of AnythingLLM:
- HTTPS with automatic TLS certificates
- HTTP Basic Auth at proxy level
- AnythingLLM not exposed directly (internal only)

Steps:

1. Create env file:

docker compose up

Security notes:
- Keep AnythingLLM built-in auth enabled (multi-user recommended for internet exposure).
- Disable public signup inside AnythingLLM unless explicitly needed.
- Keep `OPENAI_API_KEY` only inside server env, never in frontend code.

## 5) Automatize the vector store reload : 
Create a job to automatically update the vectore store weekly : 
CRON_TZ=Europe/Paris
0 7 * * 1 chatbot/reload_job.sh >> chatbot/reindex.log 2>&1


## Optional: notebooks

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name chatbot-venv --display-name "Python (chatbot-venv)"
jupyter notebook
```
