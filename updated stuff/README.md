# 🚀 Advanced Code RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for intelligent code search and question-answering across repositories. Features hybrid search (semantic + BM25), AST-aware chunking, cross-encoder reranking, and a beautiful Streamlit UI.

## ✨ Key Features

### 🎯 Advanced Retrieval
- **Hybrid Search**: Combines semantic embeddings (Jina v2) with BM25 lexical search
- **Cross-Encoder Reranking**: Uses `ms-marco-MiniLM` for precise relevance scoring
- **Query Expansion**: Automatically generates semantic variations of queries
- **RRF (Reciprocal Rank Fusion)**: Intelligently merges results from multiple retrievers

### 🧠 Intelligent Chunking
- **AST-Aware Parsing**: Extracts functions, classes, and methods as semantic units (Python)
- **Overlap Strategy**: Maintains context with configurable chunk overlap
- **Rich Metadata**: Tracks complexity, symbols, imports, docstrings, and more
- **Token-Aware**: Respects token limits for optimal embedding quality

### 🎨 Production UI
- **Real-time Search**: Sub-second query responses with progress indicators
- **Interactive Visualizations**: Language distribution, repository stats, chunk types
- **Advanced Filters**: Filter by language, repository, and adjust result counts
- **Export Options**: JSON and Markdown export for all responses
- **Query History**: Track and revisit past searches

### ⚡ Performance
- **Parallel Processing**: Multi-threaded repository ingestion
- **Batch Operations**: Efficient database operations with batching
- **Progress Tracking**: Real-time feedback with tqdm integration
- **Caching**: Smart caching of statistics and embeddings

## 📋 Prerequisites

- Python 3.9+
- GROQ API Key ([Get one here](https://console.groq.com))
- 4GB+ RAM recommended
- Git installed

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd code-rag-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Quick Start

### Launch the Web UI
```bash
streamlit run app.py
```

Then navigate to `http://localhost:8501`

### CLI Usage

#### Process a Repository
```bash
python processor.py https://github.com/username/repo --name my-repo --workers 4
```

#### Query the System
```python
from core import EnhancedRAGEngine

engine = EnhancedRAGEngine()
response = engine.query("How is authentication implemented?")
print(response['answer'])
```

## 📖 Usage Guide

### 1. Index Repositories

**Via Web UI:**
1. Go to the "📦 Repositories" tab
2. Enter a GitHub/GitLab URL
3. Click "Process Repository"
4. Wait for processing to complete

**Via CLI:**
```bash
python processor.py https://github.com/fastapi/fastapi --name fastapi --workers 8
```

### 2. Query Your Codebase

**Via Web UI:**
1. Go to "🔍 Query" tab
2. Type your question
3. Optionally apply filters (language, repository)
4. Click "Ask Assistant"

**Example Queries:**
- "How is authentication implemented?"
- "Show me database connection patterns"
- "Find error handling examples in Python"
- "Explain the API routing structure"
- "How are tests organized?"

### 3. View Results

Each query returns:
- **Answer**: AI-generated explanation based on your code
- **Source Chunks**: Relevant code snippets with:
  - File name and repository
  - Line numbers
  - Relevance score
  - Chunk type (function, class, block)
  - Complexity metrics

### 4. Export & Share

- Download results as JSON or Markdown
- Share insights with your team
- Track queries in the History tab

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Query                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Query Expansion                           │
│  (Semantic variations, synonyms, code-specific terms)        │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────────┐     ┌───────────────────┐
│  Semantic Search  │     │   BM25 Search     │
│  (Jina Embeddings)│     │  (Lexical Match)  │
└────────┬──────────┘     └────────┬──────────┘
         │                         │
         └────────────┬────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Reciprocal Rank      │
         │   Fusion (RRF)         │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │   Cross-Encoder        │
         │   Reranking            │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │   LLM Generation       │
         │   (Groq/Llama 3.1)     │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │   Formatted Response   │
         │   + Source Citations   │
         └────────────────────────┘
```

## 🧩 Components

### `core.py` - RAG Engine
- **ASTAwareChunker**: Intelligent code parsing and chunking
- **HybridRetriever**: Multi-strategy search with reranking
- **EnhancedRAGEngine**: Complete RAG pipeline orchestration

### `processor.py` - Repository Processing
- Repository cloning and validation
- Parallel file processing
- Batch database insertion
- Progress tracking

### `app.py` - Web Interface
- Streamlit-based UI
- Real-time search
- Interactive visualizations
- Export functionality

## ⚙️ Configuration

### Chunking Parameters
```python
chunker = ASTAwareChunker()
chunks = chunker.chunk_code(
    content=code,
    filepath=path,
    repo_name=name,
    language='python'
)
# Automatically uses AST for Python, falls back to line-based for others
```

### Search Parameters
```python
response = engine.query(
    query="your question",
    n_results=8,  # Number of chunks to retrieve
    filters={'language': 'python'}  # Optional filters
)
```

### Processing Parameters
```bash
python processor.py <repo_url> \
  --name custom-name \
  --workers 8 \  # Parallel workers
  --db-path data/custom_db
```

## 📊 Supported Languages

- Python (AST-aware)
- JavaScript/TypeScript
- Java
- C/C++
- Go, Rust, PHP, Ruby
- C#, Kotlin, Swift, Scala
- HTML, CSS, SQL
- Bash, YAML, JSON, Markdown

## 🎯 Performance Tips

1. **Use Parallel Workers**: Set `--workers` to match your CPU cores
2. **Filter by Language**: Use language filters for faster searches
3. **Adjust Result Count**: Lower `n_results` for faster responses
4. **Process Incrementally**: Index repos one at a time to monitor quality
5. **Clear Old Data**: Delete `data/chromadb_v2` to start fresh

## 🐛 Troubleshooting

### "No relevant code found"
- Ensure repositories are processed successfully
- Try broader queries
- Check database stats in the sidebar

### Slow Processing
- Reduce `--workers` count
- Skip large repositories (>100k lines)
- Check network connection for cloning

### Out of Memory
- Process smaller repositories
- Reduce batch size in `processor.py`
- Close other applications

### API Rate Limits
- Use smaller `max_tokens` in queries
- Add delays between batch operations
- Switch to different Groq model

## 🔒 Security & Privacy

- **Local Processing**: All data stays on your machine
- **No Code Upload**: Repositories are cloned locally
- **API Privacy**: Only queries sent to Groq, not full code
- **Secure Storage**: ChromaDB uses local persistent storage

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional language support (AST parsers)
- More embedding models
- UI enhancements
- Performance optimizations

## 📝 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

- **Groq** - Lightning-fast LLM inference
- **ChromaDB** - Vector database
- **Jina AI** - Code embeddings
- **Streamlit** - Beautiful UI framework

## 📧 Support

For issues, questions, or feedback:
1. Check existing documentation
2. Review troubleshooting section
3. Open a GitHub issue with details

---

Built with ❤️ for developers who want to understand their codebases better.
