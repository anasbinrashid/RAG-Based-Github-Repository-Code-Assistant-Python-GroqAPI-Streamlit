# ⚡ Quick Start Guide

Get up and running in 5 minutes!

## 🚀 Installation (2 minutes)

```bash
# 1. Clone and navigate
cd code-rag-system

# 2. Run setup script (creates venv, installs deps)
chmod +x setup.sh
./setup.sh

# 3. Activate environment
source venv/bin/activate

# 4. Add your API key
# Edit .env and replace 'your_groq_api_key_here' with your actual key
nano .env
```

## 🎯 First Steps (3 minutes)

### Option A: Web Interface (Recommended)

```bash
# Launch the beautiful UI
streamlit run app.py
```

Then:
1. Go to http://localhost:8501
2. Click "📦 Repositories" tab
3. Enter: `https://github.com/pallets/click`
4. Click "Process Repository" (takes ~1 minute)
5. Go to "🔍 Query" tab
6. Ask: "How does Click parse command line arguments?"
7. Get instant, context-aware answers! 🎉

### Option B: Command Line

```bash
# Interactive CLI
python cli.py

# At the prompt, type:
> process https://github.com/pallets/click

# Then query:
> How does Click handle command options?
```

### Option C: Python API

```python
from core import EnhancedRAGEngine
from processor import RepositoryProcessor

# Initialize
engine = EnhancedRAGEngine()
processor = RepositoryProcessor(engine)

# Index a repo
processor.process_repository("https://github.com/pallets/click")

# Query
response = engine.query("How does Click parse arguments?")
print(response['answer'])
```

## 🎓 Example Queries

Try these on any indexed codebase:

**Understanding Code:**
- "How is authentication implemented?"
- "Explain the database connection logic"
- "What design patterns are used?"

**Finding Code:**
- "Show me error handling examples"
- "Find all API endpoints"
- "Where are the database models defined?"

**Implementation Help:**
- "How to add a new API endpoint?"
- "How is logging configured?"
- "Show me test examples"

## 📊 Check Your Setup

```bash
# View statistics
python cli.py --stats

# Should show:
# Total chunks: XXX
# Languages: python, javascript, etc.
# Repositories: click, etc.
```

## 🆘 Troubleshooting

**"GROQ_API_KEY required"**
- Edit `.env` and add your key from https://console.groq.com

**"No relevant code found"**
- Make sure you processed a repository first
- Try broader queries

**Slow processing**
- Normal for large repos (1-5 min for medium repos)
- Use `--workers 8` for faster processing

**Import errors**
- Run: `pip install -r requirements.txt`
- Make sure venv is activated

## 🎨 UI Overview

**Query Tab:**
- Ask questions about your code
- Get AI-generated answers with source citations
- Export as JSON or Markdown

**Repositories Tab:**
- Add GitHub/GitLab repositories
- View indexed repos and stats
- Process multiple repos

**History Tab:**
- View past queries
- Track performance metrics
- Success rate monitoring

## 🔥 Pro Tips

1. **Start with small repos** (< 50 files) to test
2. **Use filters** (language, repo) for faster searches
3. **Try example queries** in the UI's "Example Queries" dropdown
4. **Export important answers** for documentation
5. **Check stats regularly** to monitor growth

## 📚 Next Steps

Once you're comfortable:

1. **Index your own repositories**
2. **Experiment with advanced filters**
3. **Try batch processing multiple repos**
4. **Explore the CLI for automation**
5. **Check out example.py for API usage**

## 🎯 Common Workflows

### Workflow 1: Understanding a New Codebase
```bash
# 1. Index the repo
python processor.py https://github.com/user/repo

# 2. Ask high-level questions
"What is the architecture of this application?"
"How is the project structured?"
"What are the main components?"

# 3. Dive deeper
"How does authentication work?"
"Show me the database schema"
"Explain the API design"
```

### Workflow 2: Finding Examples
```bash
# 1. Query for patterns
"Show me error handling examples in Python"

# 2. Filter results
# Use language: python, n_results: 15

# 3. Export for reference
# Click "Download JSON" to save examples
```

### Workflow 3: Code Review
```bash
# 1. Index PR/branch
python processor.py https://github.com/user/repo --name feature-branch

# 2. Ask review questions
"Are there any security issues?"
"Is error handling comprehensive?"
"Are tests included?"
```

## 🏆 Success Checklist

- [ ] Setup completed without errors
- [ ] Processed at least one repository
- [ ] Successfully queried and got results
- [ ] Explored the UI tabs
- [ ] Tried filtering by language
- [ ] Exported a query result
- [ ] Checked database statistics

**All checked?** You're ready to use the system productively! 🎉

---

Need help? Check:
- README.md for full documentation
- IMPROVEMENTS.md to understand the system
- example.py for code samples
