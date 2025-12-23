#!/usr/bin/env python3
"""
Example Usage and Demo Script
Demonstrates all features of the Code RAG system
"""

import os
from dotenv import load_dotenv
from core import EnhancedRAGEngine
from processor import RepositoryProcessor

load_dotenv()


def demo_basic_query():
    """Basic query example"""
    print("=" * 60)
    print("DEMO: Basic Query")
    print("=" * 60)
    
    engine = EnhancedRAGEngine()
    
    query = "How is authentication implemented?"
    print(f"\nQuery: {query}")
    
    response = engine.query(query)
    
    if response['success']:
        print(f"\nAnswer:\n{response['answer']}")
        print(f"\nSources: {len(response['chunks'])}")
        print(f"Time: {response['response_time']:.2f}s")
    else:
        print(f"Error: {response['answer']}")


def demo_filtered_search():
    """Filtered search example"""
    print("\n" + "=" * 60)
    print("DEMO: Filtered Search")
    print("=" * 60)
    
    engine = EnhancedRAGEngine()
    
    # Search only Python files
    query = "Show me error handling patterns"
    filters = {'language': 'python'}
    
    print(f"\nQuery: {query}")
    print(f"Filters: {filters}")
    
    response = engine.query(query, n_results=5, filters=filters)
    
    if response['success']:
        print(f"\nFound {len(response['chunks'])} Python chunks")
        for i, chunk in enumerate(response['chunks'][:3], 1):
            meta = chunk['metadata']
            print(f"\n{i}. {meta['filename']} (Lines {meta['start_line']}-{meta['end_line']})")
            print(f"   Type: {meta['chunk_type']}, Score: {chunk.get('score', 0):.3f}")


def demo_repository_processing():
    """Repository processing example"""
    print("\n" + "=" * 60)
    print("DEMO: Repository Processing")
    print("=" * 60)
    
    engine = EnhancedRAGEngine()
    processor = RepositoryProcessor(engine)
    
    # Example: Process a small public repo
    repo_url = "https://github.com/pallets/click"  # Small, well-structured repo
    
    print(f"\nProcessing: {repo_url}")
    print("This may take a few minutes...")
    
    result = processor.process_repository(repo_url, max_workers=4)
    
    if result['success']:
        print(f"\n✓ Success!")
        print(f"  Repository: {result['repo_name']}")
        print(f"  Files processed: {result['files_processed']}")
        print(f"  Chunks created: {result['chunks_created']}")
        
        # Now query the newly indexed repo
        print("\nTesting query on newly indexed repo...")
        response = engine.query("How does Click handle command options?")
        
        if response['success']:
            print(f"\nAnswer:\n{response['answer'][:300]}...")
    else:
        print(f"✗ Failed: {result['error']}")


def demo_advanced_features():
    """Advanced features demonstration"""
    print("\n" + "=" * 60)
    print("DEMO: Advanced Features")
    print("=" * 60)
    
    engine = EnhancedRAGEngine()
    
    # 1. Query expansion
    print("\n1. Query Expansion:")
    query = "How to implement authentication?"
    expanded = engine._expand_query(query)
    print(f"   Original: {query}")
    print(f"   Expanded: {expanded}")
    
    # 2. Hybrid search demonstration
    print("\n2. Hybrid Search (Semantic + BM25):")
    chunks = engine.retriever.search("database connection", n_results=5)
    print(f"   Found {len(chunks)} chunks using hybrid retrieval")
    for chunk in chunks[:3]:
        print(f"   - {chunk['metadata']['filename']}: {chunk.get('score', 0):.3f}")
    
    # 3. Statistics
    print("\n3. Database Statistics:")
    stats = engine.get_stats()
    print(f"   Total chunks: {stats.get('total_chunks', 0):,}")
    print(f"   Languages: {len(stats.get('languages', {}))}")
    print(f"   Repositories: {len(stats.get('repositories', {}))}")


def demo_comparison():
    """Compare different query strategies"""
    print("\n" + "=" * 60)
    print("DEMO: Query Strategy Comparison")
    print("=" * 60)
    
    engine = EnhancedRAGEngine()
    
    query = "error handling"
    
    # Semantic only
    print(f"\nQuery: '{query}'")
    print("\n1. Semantic Search Only:")
    semantic = engine.retriever._semantic_search(query, 5, None)
    for i, chunk in enumerate(semantic[:3], 1):
        print(f"   {i}. {chunk['metadata']['filename']}: {chunk['score']:.3f}")
    
    # BM25 only
    print("\n2. BM25 Search Only:")
    bm25 = engine.retriever._bm25_search(query, 5)
    for i, chunk in enumerate(bm25[:3], 1):
        print(f"   {i}. {chunk['metadata']['filename']}: {chunk['score']:.3f}")
    
    # Hybrid (merged)
    print("\n3. Hybrid Search (RRF):")
    hybrid = engine.retriever.search(query, 5)
    for i, chunk in enumerate(hybrid[:3], 1):
        score = chunk.get('rerank_score', chunk.get('score', 0))
        print(f"   {i}. {chunk['metadata']['filename']}: {score:.3f}")


def demo_batch_queries():
    """Batch query processing"""
    print("\n" + "=" * 60)
    print("DEMO: Batch Query Processing")
    print("=" * 60)
    
    engine = EnhancedRAGEngine()
    
    queries = [
        "How is authentication implemented?",
        "Show database connection code",
        "Find error handling examples",
        "Explain the API structure",
        "How are tests organized?"
    ]
    
    print(f"\nProcessing {len(queries)} queries...")
    
    results = []
    for query in queries:
        response = engine.query(query, n_results=5)
        results.append({
            'query': query,
            'success': response['success'],
            'chunks': len(response.get('chunks', [])),
            'time': response.get('response_time', 0)
        })
    
    print("\nResults:")
    for i, result in enumerate(results, 1):
        status = "✓" if result['success'] else "✗"
        print(f"{i}. {status} {result['query'][:40]}...")
        print(f"   Chunks: {result['chunks']}, Time: {result['time']:.2f}s")
    
    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"\nAverage response time: {avg_time:.2f}s")


def demo_export():
    """Export functionality demo"""
    print("\n" + "=" * 60)
    print("DEMO: Export Functionality")
    print("=" * 60)
    
    engine = EnhancedRAGEngine()
    
    query = "How does error handling work?"
    response = engine.query(query)
    
    if response['success']:
        # Export as JSON
        import json
        from datetime import datetime
        
        export_data = {
            'query': query,
            'answer': response['answer'],
            'metadata': {
                'response_time': response['response_time'],
                'model': response['model'],
                'chunks_analyzed': len(response['chunks']),
                'timestamp': datetime.now().isoformat()
            },
            'sources': [
                {
                    'filename': c['metadata']['filename'],
                    'repo': c['metadata']['repo_name'],
                    'lines': f"{c['metadata']['start_line']}-{c['metadata']['end_line']}",
                    'score': c.get('score', 0)
                }
                for c in response['chunks']
            ]
        }
        
        filename = f"query_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✓ Exported to {filename}")
        print(f"  Query: {query}")
        print(f"  Sources: {len(response['chunks'])}")


def run_all_demos():
    """Run all demonstrations"""
    print("\n" + "=" * 60)
    print("CODE RAG SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n❌ Error: GROQ_API_KEY environment variable required!")
        print("Set it in your .env file or export GROQ_API_KEY=your_key")
        return
    
    demos = [
        ("Basic Query", demo_basic_query),
        ("Filtered Search", demo_filtered_search),
        ("Advanced Features", demo_advanced_features),
        ("Query Comparison", demo_comparison),
        ("Batch Processing", demo_batch_queries),
        ("Export Functionality", demo_export),
        # ("Repository Processing", demo_repository_processing),  # Commented out - takes time
    ]
    
    for name, demo_func in demos:
        try:
            input(f"\nPress Enter to run: {name}...")
            demo_func()
        except KeyboardInterrupt:
            print("\n\nDemo interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        demo_name = sys.argv[1].lower()
        
        demos = {
            'basic': demo_basic_query,
            'filtered': demo_filtered_search,
            'processing': demo_repository_processing,
            'advanced': demo_advanced_features,
            'comparison': demo_comparison,
            'batch': demo_batch_queries,
            'export': demo_export,
        }
        
        if demo_name in demos:
            demos[demo_name]()
        else:
            print(f"Unknown demo: {demo_name}")
            print(f"Available: {', '.join(demos.keys())}")
    else:
        run_all_demos()

