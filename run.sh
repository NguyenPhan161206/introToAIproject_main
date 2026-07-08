#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[Run] Dọn dẹp server cũ trên port 5000..."
kill $(lsof -ti:5000) 2>/dev/null || true
sleep 1

echo "[Run] Cài đặt dependencies (nếu thiếu)..."
./venv/bin/pip install -r requirements-web.txt -q

echo "[Run] Khởi động server..."
echo "[Run] Truy cập: http://127.0.0.1:5000"
./venv/bin/python3 web/app.py
