#!/usr/bin/env bash
# Rotoscope Studio one-click starter for macOS and Linux.
# This script does NOT install Python itself. It verifies
# the environment, installs missing Python packages, then
# launches the backend server.

set -e
cd "$(dirname "$0")"

PYTHON_EXE="python3"
if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
    echo "[ERROR] python3 is not installed or not on PATH."
    echo "Please install Python 3.9 or higher."
    exit 1
fi

echo "============================================"
echo "Rotoscope Studio - one-click starter"
echo "============================================"
echo ""

echo "Step 1 of 3: checking environment ..."
if ! "$PYTHON_EXE" scripts/setup_check.py; then
    echo ""
    echo "Step 2 of 3: installing missing packages ..."
    "$PYTHON_EXE" scripts/install_deps.py
else
    echo ""
    echo "Step 2 of 3: all packages already installed, skipping."
fi

echo ""
echo "Step 3 of 3: starting backend server ..."
echo "Open http://127.0.0.1:8000 in your browser."
echo ""

# Try to open the browser in the background.
( command -v xdg-open >/dev/null 2>&1 && xdg-open http://127.0.0.1:8000 ) >/dev/null 2>&1 &
( command -v open >/dev/null 2>&1 && open http://127.0.0.1:8000 ) >/dev/null 2>&1 &

"$PYTHON_EXE" -m uvicorn app.main:app --host 127.0.0.1 --port 8000