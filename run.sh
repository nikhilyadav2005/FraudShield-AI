#!/usr/bin/env bash
# Quick Start Script for Credit Card Fraud Detection System

echo "=========================================================="
echo "🛡️ Starting Credit Card Fraud Detection System..."
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python3"
    STREAMLIT_BIN=".venv/bin/streamlit"
else
    PYTHON_BIN="python3"
    STREAMLIT_BIN="streamlit"
fi

# Check if database or sample data exists, otherwise initialize
if [ ! -f "data/sample_transactions.csv" ] || [ ! -f "data/fraud_system.db" ]; then
    echo "Initializing sample data and baseline models..."
    PYTHONPATH=. $PYTHON_BIN src/initialize_system.py
fi

echo "Launching Streamlit Dashboard..."
PYTHONPATH=. $STREAMLIT_BIN run app.py --server.port=8501 --server.headless=true
