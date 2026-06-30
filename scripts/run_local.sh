#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv backend/.venv
# shellcheck disable=SC1091
source backend/.venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r backend/requirements.txt

[ -f backend/.env ] || cp .env.example backend/.env

echo "Pathfinder → http://localhost:8000  (open on your phone over the same network for camera + haptics)"
exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
