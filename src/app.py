#!/usr/bin/env python3

import streamlit as st
import os
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import threading
from queue import Queue
import git
from urllib.parse import urlparse
import re
import logging

# Suppress Streamlit telemetry
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Import our actual classes
try:
    from agent import CodeAgent
    from chunker_embedder import GitHubProcessor
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Code Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 4px;
        text-align: center;
    }
    
    .source-info {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state with minimal required variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processing_queue' not in st.session_state:
        st.session_state.processing_queue = Queue()
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None
    if 'stats' not in st.session_state:
        st.session_state.stats = {}

def load_agent():
    """Initialize agent with error handling"""
    if st.session_state.agent is None:
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY environment variable required!")
            return False
        
        try:
            with st.spinner("Initializing agent..."):
                st.session_state.agent = CodeAgent(
                    db_path="data/chromadb",
                    model="llama3-70b-8192"
                )
                st.success("Agent initialized successfully!")
                return True
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            return False
    return True

def is_valid_git_url(url: str) -> bool:
    """Validate git repository URL"""
    patterns = [
        r'^https://github\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?/?$',
        r'^https://gitlab\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?/?$',
        r'^git@github\.com:[\w\-\.]+/[\w\-\.]+\.git$'
    ]
    return any(re.match(pattern, url.strip()) for pattern in patterns)

def extract_repo_name(url: str) -> str:
    """Extract repository name from URL"""
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if path.endswith('.git'):
        path = path[:-4]
    return path.split('/')[-1] if '/' in path else path

def process_repository_thread(repo_url: str, progress_queue: Queue):
    """Process repository in background thread"""
    try:
        repo_name = extract_repo_name(repo_url)
        progress_queue.put({"status": "cloning", "message": f"Cloning {repo_name}..."})
        
        processor = GitHubProcessor(base_dir="data")
        result = processor.process_repository(repo_url, repo_name)
        
        if result['success']:
            progress_queue.put({
                "status": "success",
                "message": f"Successfully processed {repo_name}",
                "stats": {
                    "files_processed": result['files_processed'],
                    "chunks_created": result['chunks_created']
                }
            })
        else:
            progress_queue.put({"status": "error", "message": result['error']})
            
    except Exception as e:
        progress_queue.put({"status": "error", "message": str(e)})

def render_sidebar():
    """Render simplified sidebar"""
    with st.sidebar:
        st.header("Repository Info")
        
        # Load and display stats
        if st.session_state.agent and st.button("Refresh Stats"):
            try:
                st.session_state.stats = st.session_state.agent.get_stats()
            except Exception as e:
                st.error(f"Failed to load stats: {e}")
        
        if st.session_state.stats:
            stats = st.session_state.stats
            st.metric("Total Chunks", stats.get('total_chunks', 0))
            
            # Languages
            languages = stats.get('languages', {})
            if languages:
                st.subheader("Languages")
                for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.write(f"• {lang}: {count}")
            
            # Repositories
            repos = stats.get('repositories', {})
            if repos:
                st.subheader("Repositories")
                for repo, count in repos.items():
                    st.write(f"• {repo}: {count} chunks")

def render_query_interface():
    """Main query interface"""
    st.header("Code Assistant")
    st.write("Ask questions about your codebase")
    
    # Example queries
    with st.expander("Example Queries"):
        examples = [
            "Explain how authentication works",
            "Find Python error handling examples", 
            "Show me database connection patterns",
            "How is logging implemented?"
        ]
        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}"):
                st.session_state.example_query = example
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        value=st.session_state.get('example_query', ''),
        height=100,
        placeholder="e.g., How does authentication work in this codebase?"
    )
    
    # Clear example query after using it
    if 'example_query' in st.session_state:
        del st.session_state.example_query
    
    # Submit button
    if st.button("Ask Assistant", type="primary"):
        if query.strip():
            if not load_agent():
                return
            
            with st.spinner("Processing query..."):
                try:
                    response = st.session_state.agent.query(query.strip())
                    
                    if response['success']:
                        # Add to history
                        st.session_state.conversation_history.append({
                            'query': query,
                            'response': response,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        # Keep only last 20 conversations
                        if len(st.session_state.conversation_history) > 20:
                            st.session_state.conversation_history = st.session_state.conversation_history[-20:]
                        
                        # Display response
                        render_response(response, query)
                    else:
                        st.error(f"Query failed: {response.get('answer', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a query")

def render_response(response: dict, query: str):
    """Display query response"""
    st.subheader("Answer")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><strong>Response Time</strong><br>{response.get("response_time", 0):.2f}s</div>', 
                   unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><strong>Model</strong><br>{response.get("model_used", "Unknown")}</div>', 
                   unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><strong>Sources</strong><br>{len(response.get("sources", []))}</div>', 
                   unsafe_allow_html=True)
    
    # Answer
    st.markdown(response.get('answer', 'No answer provided'))
    
    # Sources
    sources = response.get('sources', [])
    if sources:
        st.subheader("Sources")
        
        # Sources table
        source_data = []
        for i, source in enumerate(sources, 1):
            source_data.append({
                "#": i,
                "File": source.get('filename', 'Unknown'),
                "Repository": source.get('repository', 'Unknown'),
                "Language": source.get('language', 'Unknown'),
                "Lines": source.get('lines', 'Unknown'),
                "Relevance": f"{source.get('relevance_score', 0):.3f}"
            })
        
        if source_data:
            df = pd.DataFrame(source_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        # JSON export
        export_data = {
            'query': query,
            'answer': response.get('answer', ''),
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        }
        st.download_button(
            "Download JSON",
            json.dumps(export_data, indent=2),
            file_name=f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Markdown export
        markdown_content = f"""# Query Response

## Query
{query}

## Answer
{response.get('answer', '')}

## Sources ({len(sources)})
"""
        for i, source in enumerate(sources, 1):
            markdown_content += f"{i}. {source.get('filename', 'Unknown')} ({source.get('language', 'Unknown')})\n"
        
        st.download_button(
            "Download Markdown",
            markdown_content,
            file_name=f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

def render_repository_manager():
    """Repository management interface"""
    st.header("Repository Manager")
    
    # Add repository
    repo_url = st.text_input(
        "Repository URL",
        placeholder="https://github.com/username/repository-name"
    )
    
    if repo_url and not is_valid_git_url(repo_url):
        st.error("Invalid repository URL")
        return
    
    if st.button("Process Repository", disabled=not repo_url):
        # Clear previous processing status
        while not st.session_state.processing_queue.empty():
            try:
                st.session_state.processing_queue.get_nowait()
            except:
                break
        
        # Start processing
        thread = threading.Thread(
            target=process_repository_thread,
            args=(repo_url, st.session_state.processing_queue)
        )
        thread.daemon = True
        thread.start()
        st.session_state.processing_status = "running"
        st.rerun()
    
    # Show processing status
    if st.session_state.processing_status == "running":
        try:
            while not st.session_state.processing_queue.empty():
                update = st.session_state.processing_queue.get_nowait()
                
                if update["status"] == "cloning":
                    st.info(update["message"])
                elif update["status"] == "success":
                    st.success(update["message"])
                    if 'stats' in update:
                        stats = update['stats']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Files Processed", stats['files_processed'])
                        with col2:
                            st.metric("Chunks Created", stats['chunks_created'])
                    st.session_state.processing_status = "completed"
                    st.session_state.agent = None  # Force reload
                elif update["status"] == "error":
                    st.error(f"Error: {update['message']}")
                    st.session_state.processing_status = "error"
        except:
            pass
        
        if st.session_state.processing_status == "running":
            st.info("Processing repository... This may take a few minutes.")
            time.sleep(2)
            st.rerun()
    
    # Show existing repositories
    if st.session_state.stats:
        repos = st.session_state.stats.get('repositories', {})
        if repos:
            st.subheader("Existing Repositories")
            repo_data = [{"Repository": name, "Chunks": count} for name, count in repos.items()]
            st.dataframe(pd.DataFrame(repo_data), hide_index=True)

def render_conversation_history():
    """Show conversation history"""
    st.header("Conversation History")
    
    if not st.session_state.conversation_history:
        st.info("No conversations yet")
        return
    
    for i, item in enumerate(reversed(st.session_state.conversation_history)):
        with st.expander(f"Query {len(st.session_state.conversation_history) - i}: {item['query'][:50]}... ({item['timestamp']})"):
            st.write(f"**Query:** {item['query']}")
            st.write(f"**Answer:** {item['response'].get('answer', '')}")
            
            sources = item['response'].get('sources', [])
            if sources:
                st.write(f"**Sources:** {len(sources)} files referenced")

def main():
    """Main application"""
    init_session_state()
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY environment variable is required!")
        st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Query Assistant", "Repository Manager", "History"])
    
    with tab1:
        render_query_interface()
    
    with tab2:
        render_repository_manager()
    
    with tab3:
        render_conversation_history()

if __name__ == "__main__":
    main()