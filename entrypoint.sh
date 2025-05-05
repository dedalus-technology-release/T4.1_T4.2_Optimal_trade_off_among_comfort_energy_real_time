#!/bin/bash

set -e

PORT=${APP_PORT:-8000}

echo "[*] Waiting for DB init..."

if [ ! -f "data/users.db" ]; then
  echo "[*] First run detected: initializing admin user..."
  python create_admin.py
else
  echo "[*] Existing DB found, skipping admin creation."
fi

echo "[*] Starting FastAPI server..."
exec uvicorn service:app --host 0.0.0.0 --port "$PORT"