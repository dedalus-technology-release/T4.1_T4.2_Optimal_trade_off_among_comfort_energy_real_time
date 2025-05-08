#!/bin/bash

set -e

PORT=${APP_PORT:-8000}
DB_PATH=${DB_PATH:-/app/data/users.db}

echo "[*] Waiting for DB init..."

if [ ! -f "$DB_PATH" ]; then
  echo "[*] First run detected: initializing admin user..."
  export DB_PATH="$DB_PATH"
  python create_admin.py
else
  echo "[*] Existing DB found, skipping admin creation."
fi

echo "[*] Starting FastAPI server..."
exec uvicorn service:app --host 0.0.0.0 --port "$PORT"
