#!/usr/bin/env python3
"""
Production Streamlit UI for Code RAG System
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

# Ensure src directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core import EnhancedRAGEngine
from processor import RepositoryProcessor

# Suppress telemetry
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Page config
st.set_page_config(
    page_title="Code RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .chunk-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .chunk-meta {
        display: flex;
        gap: 1rem;
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    
    .score-badge {
        background: #667eea;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    defaults = {
        'engine': None,
        'processor': None,
        'conversation_history': [],
        'processing_status': None,
        'stats_cache': None,
        'stats_last_updated': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_engine():
    """Load RAG engine"""
    if st.session_state.engine is None:
        if not os.getenv("GROQ_API_KEY"):
            st.error("❌ GROQ_API_KEY environment variable required!")
            st.stop()
        
        try:
            with st.spinner("🔧 Initializing RAG engine..."):
                st.session_state.engine = EnhancedRAGEngine()
                st.session_state.processor = RepositoryProcessor(st.session_state.engine)
                st.success("✅ Engine initialized successfully!")
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            st.stop()


def load_stats(force_refresh: bool = False):
    """Load stats with caching"""
    if force_refresh or st.session_state.stats_cache is None:
        if st.session_state.engine:
            st.session_state.stats_cache = st.session_state.engine.get_stats()
            st.session_state.stats_last_updated = datetime.now()
    
    return st.session_state.stats_cache


def render_sidebar():
    """Enhanced sidebar"""
    with st.sidebar:
        st.markdown("### 📊 Database Statistics")
        
        if st.button("🔄 Refresh Stats", use_container_width=True):
            load_stats(force_refresh=True)
        
        stats = load_stats()
        
        if stats and 'error' not in stats:
            # Total chunks metric
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Total Chunks</div>
                    <div class="metric-value">{stats.get('total_chunks', 0):,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Repositories</div>
                    <div class="metric-value">{len(stats.get('repositories', {}))}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Languages breakdown
            languages = stats.get('languages', {})
            if languages:
                st.markdown("#### 💻 Languages")
                lang_df = pd.DataFrame([
                    {'Language': lang.title(), 'Chunks': count}
                    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:8]
                ])
                
                fig = px.bar(lang_df, x='Chunks', y='Language', orientation='h',
                           color='Chunks', color_continuous_scale='Viridis')
                fig.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            # Repositories
            repos = stats.get('repositories', {})
            if repos:
                st.markdown("#### 📦 Repositories")
                for repo, count in list(repos.items())[:5]:
                    st.markdown(f"**{repo}**")
                    st.progress(count / stats['total_chunks'])
                    st.caption(f"{count:,} chunks")
            
            # Chunk types
            chunk_types = stats.get('chunk_types', {})
            if chunk_types:
                st.markdown("#### 🧩 Chunk Types")
                for chunk_type, count in sorted(chunk_types.items(), key=lambda x: x[1], reverse=True):
                    pct = (count / stats['total_chunks']) * 100
                    st.metric(chunk_type.title(), f"{count:,}", f"{pct:.1f}%")
        
        else:
            st.info("No data available. Process a repository to get started!")
        
        # Last updated
        if st.session_state.stats_last_updated:
            st.caption(f"Updated: {st.session_state.stats_last_updated.strftime('%H:%M:%S')}")


def render_query_tab():
    """Query interface"""
    st.markdown('<h1 class="main-header">🔍 Code Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions about your codebase using advanced semantic search")
    
    # Example queries
    with st.expander("💡 Example Queries", expanded=False):
        examples = [
            "How is authentication implemented?",
            "Show me error handling patterns",
            "Find database connection logic",
            "Explain the API routing structure",
            "How are tests organized?",
            "Show me utility functions for data processing"
        ]
        
        cols = st.columns(3)
        for idx, example in enumerate(examples):
            with cols[idx % 3]:
                if st.button(example, key=f"ex_{idx}", use_container_width=True):
                    st.session_state.query_input = example
    
    # Advanced filters
    with st.expander("⚙️ Advanced Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            language_filter = st.selectbox(
                "Language",
                ["All"] + list(load_stats().get('languages', {}).keys()) if load_stats() else ["All"]
            )
        
        with col2:
            repo_filter = st.selectbox(
                "Repository",
                ["All"] + list(load_stats().get('repositories', {}).keys()) if load_stats() else ["All"]
            )
        
        with col3:
            n_results = st.slider("Results", 3, 15, 8)
    
    # Query input
    query = st.text_area(
        "🎯 Your Question:",
        value=st.session_state.get('query_input', ''),
        height=120,
        placeholder="e.g., How is authentication handled in the API endpoints?"
    )
    
    # Clear cached query
    if 'query_input' in st.session_state:
        del st.session_state.query_input
    
    # Query button
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("🚀 Ask Assistant", type="primary", use_container_width=True)
    with col2:
        clear = st.button("🗑️ Clear", use_container_width=True)
    
    if clear:
        st.session_state.conversation_history = []
        st.rerun()
    
    if submit and query.strip():
        load_engine()
        
        # Build filters
        filters = {}
        if language_filter != "All":
            filters['language'] = language_filter
        if repo_filter != "All":
            filters['repo_name'] = repo_filter
        
        with st.spinner("🔍 Searching codebase..."):
            response = st.session_state.engine.query(
                query.strip(),
                n_results=n_results,
                filters=filters if filters else None
            )
        
        if response['success']:
            # Add to history
            st.session_state.conversation_history.insert(0, {
                'query': query,
                'response': response,
                'timestamp': datetime.now()
            })
            
            # Keep last 50
            st.session_state.conversation_history = st.session_state.conversation_history[:50]
            
            render_response(response, query)
        else:
            st.error(f"❌ {response.get('answer', 'Query failed')}")


def render_response(response: Dict, query: str):
    """Display query response"""
    st.markdown("---")
    st.markdown("### 💬 Answer")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⏱️ Response Time", f"{response.get('response_time', 0):.2f}s")
    with col2:
        st.metric("📄 Chunks Analyzed", len(response.get('chunks', [])))
    with col3:
        st.metric("🤖 Model", response.get('model', 'unknown'))
    with col4:
        avg_score = sum(c.get('score', 0) for c in response.get('chunks', [])) / max(len(response.get('chunks', [])), 1)
        st.metric("📊 Avg Relevance", f"{avg_score:.2%}")
    
    # Answer
    st.markdown(response.get('answer', 'No answer provided'))
    
    # Source chunks
    chunks = response.get('chunks', [])
    if chunks:
        st.markdown("### 📚 Source Code")
        
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk['metadata']
            score = chunk.get('rerank_score', chunk.get('score', 0))
            
            with st.expander(
                f"**{idx}. {meta.get('filename', 'unknown')}** "
                f"({meta.get('language', 'unknown')}) - "
                f"Lines {meta.get('start_line', '?')}-{meta.get('end_line', '?')} "
                f"• Score: {score:.3f}",
                expanded=idx <= 3
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="chunk-meta">
                        <span>📦 {meta.get('repo_name', 'unknown')}</span>
                        <span>📁 {meta.get('filepath', 'unknown')}</span>
                        <span>🧩 {meta.get('chunk_type', 'block')}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'<div class="score-badge">Score: {score:.3f}</div>', unsafe_allow_html=True)
                
                # Code
                st.code(chunk['content'], language=meta.get('language', ''))
                
                # Metadata
                if meta.get('symbols'):
                    st.caption(f"🏷️ Symbols: {meta['symbols']}")
                if meta.get('complexity'):
                    st.caption(f"🔢 Complexity: {meta['complexity']}")
    
    # Export options
    st.markdown("### 💾 Export")
    col1, col2 = st.columns(2)
    
    with col1:
        export_json = {
            'query': query,
            'answer': response.get('answer', ''),
            'metadata': {
                'response_time': response.get('response_time', 0),
                'model': response.get('model', ''),
                'chunks_count': len(chunks),
                'timestamp': datetime.now().isoformat()
            },
            'sources': [
                {
                    'filename': c['metadata'].get('filename'),
                    'repo': c['metadata'].get('repo_name'),
                    'lines': f"{c['metadata'].get('start_line')}-{c['metadata'].get('end_line')}",
                    'score': c.get('score', 0)
                } for c in chunks
            ]
        }
        
        st.download_button(
            "📥 Download JSON",
            json.dumps(export_json, indent=2),
            f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            use_container_width=True
        )
    
    with col2:
        markdown = f"""# Query Response

**Query:** {query}

**Model:** {response.get('model', '')}  
**Response Time:** {response.get('response_time', 0):.2f}s

## Answer

{response.get('answer', '')}

## Sources ({len(chunks)})

"""
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk['metadata']
            markdown += f"{idx}. **{meta.get('filename')}** ({meta.get('language')}) - Lines {meta.get('start_line')}-{meta.get('end_line')}\n"
        
        st.download_button(
            "📥 Download Markdown",
            markdown,
            f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            "text/markdown",
            use_container_width=True
        )


def render_repo_tab():
    """Repository management"""
    st.markdown('<h1 class="main-header">📦 Repository Manager</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        repo_url = st.text_input(
            "Repository URL",
            placeholder="https://github.com/username/repository",
            help="Enter a GitHub/GitLab repository URL"
        )
    
    with col2:
        repo_name = st.text_input(
            "Name (optional)",
            placeholder="my-repo"
        )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_workers = st.number_input("Parallel Workers", 1, 8, 4)
    
    if st.button("🚀 Process Repository", type="primary", disabled=not repo_url, use_container_width=True):
        load_engine()
        
        with st.spinner(f"⏳ Processing {repo_url}..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            result = st.session_state.processor.process_repository(
                repo_url,
                repo_name,
                max_workers=max_workers
            )
            
            progress_bar.progress(100)
            
            if result['success']:
                st.success(f"✅ Successfully processed **{result['repo_name']}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Files Processed", result['files_processed'])
                with col2:
                    st.metric("🧩 Chunks Created", result['chunks_created'])
                
                # Force stats refresh
                load_stats(force_refresh=True)
            else:
                st.error(f"❌ Processing failed: {result['error']}")
    
    # Show existing repos
    st.markdown("---")
    st.markdown("### 📚 Indexed Repositories")
    
    stats = load_stats()
    repos = stats.get('repositories', {}) if stats else {}
    
    if repos:
        repo_data = []
        for name, count in repos.items():
            repo_data.append({
                'Repository': name,
                'Chunks': count,
                'Percentage': f"{(count / stats['total_chunks']) * 100:.1f}%"
            })
        
        df = pd.DataFrame(repo_data).sort_values('Chunks', ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No repositories indexed yet. Add one above to get started!")


def render_history_tab():
    """Conversation history"""
    st.markdown('<h1 class="main-header">📜 Query History</h1>', unsafe_allow_html=True)
    
    history = st.session_state.conversation_history
    
    if not history:
        st.info("No queries yet. Try asking a question in the Query tab!")
        return
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries", len(history))
    with col2:
        avg_time = sum(h['response'].get('response_time', 0) for h in history) / len(history)
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
    with col3:
        successful = sum(1 for h in history if h['response'].get('success', False))
        st.metric("Success Rate", f"{(successful/len(history))*100:.0f}%")
    
    st.markdown("---")
    
    # Display history
    for idx, item in enumerate(history):
        with st.expander(
            f"**{item['timestamp'].strftime('%H:%M:%S')}** • {item['query'][:80]}...",
            expanded=idx == 0
        ):
            st.markdown(f"**Query:** {item['query']}")
            st.markdown(f"**Answer:** {item['response'].get('answer', 'N/A')[:500]}...")
            
            chunks = item['response'].get('chunks', [])
            if chunks:
                st.caption(f"📚 {len(chunks)} source chunks referenced")


def main():
    """Main app"""
    init_session_state()
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("❌ GROQ_API_KEY environment variable required!")
        st.info("Set it in your .env file or environment")
        st.stop()
    
    # Sidebar
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Query", "📦 Repositories", "📜 History"])
    
    with tab1:
        render_query_tab()
    
    with tab2:
        render_repo_tab()
    
    with tab3:
        render_history_tab()


if __name__ == "__main__":
    main()
