#!/bin/bash

# Customer Support RAG Assistant - Quick Start Script
# This script helps you get the system running quickly

set -e

echo "🚀 Customer Support RAG Assistant - Quick Start"
echo "================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.11+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs store evaluation data

# Check if data needs to be set up
if [ ! -f "store/qa_texts_10k.jsonl" ] || [ ! -f "store/faiss.index" ]; then
    echo "🔄 Setting up data and indices..."
    python scripts/setup.py --all --limit 10000
else
    echo "✅ Data and indices already exist"
fi

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Check if API server is already running
if check_port 8000; then
    echo "⚠️  Port 8000 is already in use. The API server might already be running."
    echo "   You can access it at: http://localhost:8000"
else
    echo "🌐 Starting API server..."
    # Start API server in background
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    echo "   API server started with PID: $API_PID"
    echo "   API available at: http://localhost:8000"
    echo "   API docs at: http://localhost:8000/docs"
fi

# Wait a moment for API to start
sleep 3

# Test API health
echo "🏥 Testing API health..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is healthy!"
else
    echo "⚠️  API health check failed. It might still be starting up."
fi

# Check if Streamlit is already running
if check_port 8501; then
    echo "⚠️  Port 8501 is already in use. Streamlit might already be running."
    echo "   You can access it at: http://localhost:8501"
else
    echo "🎨 Starting Streamlit UI..."
    # Start Streamlit in background
    streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
    STREAMLIT_PID=$!
    echo "   Streamlit started with PID: $STREAMLIT_PID"
    echo "   UI available at: http://localhost:8501"
fi

echo ""
echo "🎉 Setup complete!"
echo "================================================"
echo "📊 Access Points:"
echo "   • Streamlit UI: http://localhost:8501"
echo "   • API Server: http://localhost:8000"
echo "   • API Docs: http://localhost:8000/docs"
echo ""
echo "🧪 Test the system:"
echo "   curl -X POST http://localhost:8000/generate_response \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"query\": \"I need help with my order\"}'"
echo ""
echo "📈 Run evaluation:"
echo "   python scripts/run_evaluation.py"
echo ""
echo "🛑 To stop the services:"
echo "   pkill -f uvicorn"
echo "   pkill -f streamlit"
echo "================================================"

# Keep script running to show logs (optional)
if [ "$1" = "--follow-logs" ]; then
    echo "📝 Following logs (Ctrl+C to exit)..."
    tail -f logs/app.log 2>/dev/null || echo "No logs yet. Start using the system to see logs."
fi
