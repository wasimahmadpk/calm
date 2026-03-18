#!/usr/bin/env bash
# Run CALM locally. Use a venv to avoid NumPy/pandas conflicts with Anaconda.
set -e
cd "$(dirname "$0")"

if [ -d .venv ]; then
  source .venv/bin/activate
else
  echo "No .venv found. Create one with:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  echo "Then add your key: cp .env.example .env && edit .env (set OPENAI_API_KEY)"
  echo "Run again: ./run_local.sh \"Load sachs and run PC\""
  exit 1
fi

if [ -z "$1" ]; then
  echo "Usage: ./run_local.sh \"Your question or request\""
  echo "Example: ./run_local.sh \"Load the sachs dataset and run PC causal discovery\""
  exit 1
fi

python run_agent.py "$@"
