#!/bin/bash

# Code RAG System - Quick Setup Script
# This script sets up the entire environment

set -e

echo "🚀 Code RAG System - Setup Script"
echo "=================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "❌ Python 3.9+ required"
    exit 1
fi
echo "✓ Python version OK"
echo ""

# Check Git
echo "📋 Checking Git..."
if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Please install Git."
    exit 1
fi
echo "✓ Git OK"
echo ""

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip -q
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q
echo "✓ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔑 Creating .env file..."
    cat > .env << EOF
# Groq API Key
# Get one at: https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here
EOF
    echo "✓ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your GROQ_API_KEY"
else
    echo "✓ .env file already exists"
fi
echo ""

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/repos
mkdir -p data/chromadb_v2
echo "✓ Directories created"
echo ""

# Test installation
echo "🧪 Testing installation..."
python3 -c "
from core import EnhancedRAGEngine
print('✓ Core module OK')
from processor import RepositoryProcessor
print('✓ Processor module OK')
import streamlit
print('✓ Streamlit OK')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ All modules imported successfully"
else
    echo "❌ Module import test failed"
    exit 1
fi
echo ""

echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your GROQ_API_KEY"
echo "2. Activate the environment: source venv/bin/activate"
echo "3. Run the web UI: streamlit run app.py"
echo "4. Or use CLI: python cli.py"
echo ""
echo "For help, see README.md"
