#!/usr/bin/env python3
"""Test script to verify the setup is working correctly"""

import os
import sys

# Set a dummy API key for testing (we're just testing imports and DB connection)
os.environ['GROQ_API_KEY'] = 'test_key_for_setup_verification'

def test_imports():
    """Test that all modules import correctly"""
    print("Testing imports...")
    
    try:
        from core import EnhancedRAGEngine, CodeChunk, ASTAwareChunker, HybridRetriever
        print("  [OK] core.py imports OK")
    except Exception as e:
        print(f"  [FAIL] core.py import failed: {e}")
        return False
    
    try:
        from processor import RepositoryProcessor
        print("  [OK] processor.py imports OK")
    except Exception as e:
        print(f"  [FAIL] processor.py import failed: {e}")
        return False
    
    try:
        from cli import InteractiveCLI
        print("  [OK] cli.py imports OK")
    except Exception as e:
        print(f"  [FAIL] cli.py import failed: {e}")
        return False
    
    return True


def test_database_connection():
    """Test that we can connect to the existing ChromaDB"""
    print("\nTesting database connection...")
    
    try:
        from core import EnhancedRAGEngine
        engine = EnhancedRAGEngine()
        stats = engine.get_stats()
        
        print(f"  [OK] Connected to ChromaDB")
        print(f"  [OK] Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  [OK] Languages: {list(stats.get('languages', {}).keys())}")
        print(f"  [OK] Repositories: {list(stats.get('repositories', {}).keys())}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Database connection failed: {e}")
        return False


def test_chunker():
    """Test the AST-aware chunker"""
    print("\nTesting AST-aware chunker...")
    
    try:
        from core import ASTAwareChunker
        from pathlib import Path
        
        chunker = ASTAwareChunker()
        
        # Test code
        test_code = '''
def hello_world():
    """A simple hello world function"""
    print("Hello, World!")
    return True

class MyClass:
    """A sample class"""
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
'''
        
        chunks = chunker.chunk_python(test_code, Path("test.py"), "test-repo")
        
        print(f"  [OK] Chunker initialized")
        print(f"  [OK] Created {len(chunks)} chunks from test code")
        
        for chunk in chunks:
            print(f"    - {chunk.chunk_type}: {chunk.symbols[0] if chunk.symbols else 'unknown'} (lines {chunk.start_line}-{chunk.end_line})")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Chunker test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("CODE RAG SYSTEM - SETUP VERIFICATION")
    print("=" * 60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_database_connection():
        all_passed = False
    
    if not test_chunker():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED - Setup is complete!")
        print("\nNext steps:")
        print("  1. Set your GROQ_API_KEY in the .env file")
        print("  2. Run: streamlit run app.py")
        print("  3. Or run: python cli.py --interactive")
    else:
        print("[ERROR] SOME TESTS FAILED - Please check the errors above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
