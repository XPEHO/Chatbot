#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"
LOCK_DIR="/tmp/chatbot-reindex-lock"

# Prevent overlapping runs.
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[reindex] another run is already in progress"
  exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

cd "$PROJECT_DIR"

echo "[reindex] starting export + vector store reload"
docker compose -f "$COMPOSE_FILE" run --rm --no-deps --build \
  chatbot-api \
  sh -lc "python export_pages.py && python vector_store.py"
echo "[reindex] done"
