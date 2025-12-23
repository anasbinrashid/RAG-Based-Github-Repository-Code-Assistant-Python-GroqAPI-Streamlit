#!/usr/bin/env python3
"""
Advanced Code RAG System - Core Engine
Semantic + BM25 Hybrid Search with AST-aware chunking
"""

import os
import json
import hashlib
import ast
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict, Counter

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import tiktoken
from groq import Groq
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class CodeChunk:
    """Enhanced code chunk with rich metadata"""
    id: str
    content: str
    filename: str
    filepath: str
    repo_name: str
    language: str
    start_line: int
    end_line: int
    chunk_type: str  # function, class, block, document
    symbols: List[str]  # function/class names
    imports: List[str]
    docstring: Optional[str]
    complexity: int  # cyclomatic complexity estimate
    token_count: int

class ASTAwareChunker:
    """AST-based intelligent code chunking"""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_python(self, content: str, filepath: Path, repo_name: str) -> List[CodeChunk]:
        """Parse Python with AST for semantic chunks"""
        chunks = []
        
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start = node.lineno - 1
                    end = node.end_lineno
                    
                    chunk_content = '\n'.join(lines[start:end])
                    docstring = ast.get_docstring(node)
                    
                    # Extract imports and symbols
                    imports = self._extract_imports(tree)
                    symbols = [node.name]
                    
                    # Calculate complexity
                    complexity = self._calculate_complexity(node)
                    
                    chunk_type = 'class' if isinstance(node, ast.ClassDef) else 'function'
                    
                    chunk = CodeChunk(
                        id=self._generate_id(repo_name, filepath, start, end),
                        content=chunk_content,
                        filename=filepath.name,
                        filepath=str(filepath),
                        repo_name=repo_name,
                        language='python',
                        start_line=start + 1,
                        end_line=end,
                        chunk_type=chunk_type,
                        symbols=symbols,
                        imports=imports,
                        docstring=docstring,
                        complexity=complexity,
                        token_count=len(self.tokenizer.encode(chunk_content))
                    )
                    chunks.append(chunk)
            
            # If no AST nodes, fall back to line-based chunking
            if not chunks:
                chunks = self._chunk_by_lines(content, filepath, repo_name, 'python')
                
        except SyntaxError:
            chunks = self._chunk_by_lines(content, filepath, repo_name, 'python')
        
        return chunks
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Estimate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _chunk_by_lines(self, content: str, filepath: Path, repo_name: str, language: str, 
                        chunk_size: int = 100, overlap: int = 20) -> List[CodeChunk]:
        """Fallback line-based chunking with overlap"""
        lines = content.split('\n')
        chunks = []
        
        for i in range(0, len(lines), chunk_size - overlap):
            end = min(i + chunk_size, len(lines))
            chunk_content = '\n'.join(lines[i:end])
            
            if not chunk_content.strip():
                continue
            
            chunk = CodeChunk(
                id=self._generate_id(repo_name, filepath, i, end),
                content=chunk_content,
                filename=filepath.name,
                filepath=str(filepath),
                repo_name=repo_name,
                language=language,
                start_line=i + 1,
                end_line=end,
                chunk_type='block',
                symbols=[],
                imports=[],
                docstring=None,
                complexity=0,
                token_count=len(self.tokenizer.encode(chunk_content))
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_code(self, content: str, filepath: Path, repo_name: str, language: str) -> List[CodeChunk]:
        """Main chunking dispatcher"""
        if language == 'python':
            return self.chunk_python(content, filepath, repo_name)
        else:
            return self._chunk_by_lines(content, filepath, repo_name, language)
    
    def _generate_id(self, repo_name: str, filepath: Path, start: int, end: int) -> str:
        """Generate unique chunk ID"""
        raw = f"{repo_name}:{filepath}:{start}-{end}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class HybridRetriever:
    """Semantic + BM25 hybrid search with reranking"""
    
    def __init__(self, collection: chromadb.Collection):
        self.collection = collection
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.bm25_index = None
        self.bm25_docs = None
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from collection"""
        try:
            all_data = self.collection.get()
            if all_data['documents']:
                tokenized_docs = [doc.lower().split() for doc in all_data['documents']]
                self.bm25_index = BM25Okapi(tokenized_docs)
                self.bm25_docs = all_data
                logger.info(f"Built BM25 index with {len(all_data['documents'])} documents")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
    
    def search(self, query: str, n_results: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """Hybrid search with semantic + BM25"""
        
        # Semantic search
        semantic_results = self._semantic_search(query, n_results * 2, filters)
        
        # BM25 search
        bm25_results = self._bm25_search(query, n_results * 2)
        
        # Merge and deduplicate
        merged = self._merge_results(semantic_results, bm25_results)
        
        # Rerank top candidates
        if len(merged) > n_results:
            reranked = self._rerank(query, merged[:n_results * 2])
            return reranked[:n_results]
        
        return merged[:n_results]
    
    def _semantic_search(self, query: str, n_results: int, filters: Optional[Dict]) -> List[Dict]:
        """Semantic vector search"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters
            )
            
            formatted = []
            if results['documents'] and results['documents'][0]:
                for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                    formatted.append({
                        'content': doc,
                        'metadata': meta,
                        'score': 1 - dist,
                        'source': 'semantic'
                    })
            return formatted
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _bm25_search(self, query: str, n_results: int) -> List[Dict]:
        """BM25 lexical search"""
        if not self.bm25_index or not self.bm25_docs:
            return []
        
        try:
            tokenized_query = query.lower().split()
            scores = self.bm25_index.get_scores(tokenized_query)
            
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append({
                        'content': self.bm25_docs['documents'][idx],
                        'metadata': self.bm25_docs['metadatas'][idx],
                        'score': scores[idx],
                        'source': 'bm25'
                    })
            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _merge_results(self, semantic: List[Dict], bm25: List[Dict]) -> List[Dict]:
        """Merge and deduplicate results using RRF (Reciprocal Rank Fusion)"""
        score_dict = {}
        k = 60  # RRF constant
        
        for rank, result in enumerate(semantic, 1):
            chunk_id = result['metadata'].get('chunk_id', str(rank))
            score_dict[chunk_id] = score_dict.get(chunk_id, {'result': result, 'score': 0})
            score_dict[chunk_id]['score'] += 1 / (k + rank)
        
        for rank, result in enumerate(bm25, 1):
            chunk_id = result['metadata'].get('chunk_id', str(rank))
            if chunk_id not in score_dict:
                score_dict[chunk_id] = {'result': result, 'score': 0}
            score_dict[chunk_id]['score'] += 1 / (k + rank)
        
        merged = sorted(score_dict.values(), key=lambda x: x['score'], reverse=True)
        return [item['result'] for item in merged]
    
    def _rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank using cross-encoder"""
        try:
            pairs = [[query, c['content']] for c in candidates]
            scores = self.reranker.predict(pairs)
            
            for candidate, score in zip(candidates, scores):
                candidate['rerank_score'] = float(score)
            
            return sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return candidates


class EnhancedRAGEngine:
    """Complete RAG engine with advanced retrieval and generation"""
    
    def __init__(self, db_path: str = "data/chromadb_v2", model: str = "llama-3.1-70b-versatile"):
        self.db_path = Path(db_path)
        self.model = model
        
        # Initialize Groq
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY required")
        self.client = Groq(api_key=self.groq_api_key)
        
        # Initialize ChromaDB
        self.db_path.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="jinaai/jina-embeddings-v2-base-code"
        )
        
        try:
            self.collection = chroma_client.get_collection("code_chunks_v2", embedding_function=embedding_fn)
        except:
            self.collection = chroma_client.create_collection("code_chunks_v2", embedding_function=embedding_fn)
        
        # Initialize components
        self.chunker = ASTAwareChunker()
        self.retriever = HybridRetriever(self.collection)
        
        logger.info(f"Initialized RAG engine with {model}")
    
    def query(self, query: str, n_results: int = 8, filters: Optional[Dict] = None) -> Dict:
        """Process query with retrieval and generation"""
        start_time = datetime.now()
        
        try:
            # Expand query
            expanded_queries = self._expand_query(query)
            
            # Retrieve with all query variations
            all_chunks = []
            for q in expanded_queries:
                chunks = self.retriever.search(q, n_results=n_results, filters=filters)
                all_chunks.extend(chunks)
            
            # Deduplicate and rank
            unique_chunks = self._deduplicate_chunks(all_chunks)[:n_results]
            
            if not unique_chunks:
                return {
                    'success': False,
                    'answer': "No relevant code found. Try processing more repositories.",
                    'chunks': [],
                    'query': query
                }
            
            # Generate answer
            answer = self._generate_answer(query, unique_chunks)
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'query': query,
                'answer': answer,
                'chunks': unique_chunks,
                'response_time': response_time,
                'model': self.model
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'success': False,
                'answer': f"Error: {str(e)}",
                'chunks': [],
                'query': query
            }
    
    def _expand_query(self, query: str) -> List[str]:
        """Intelligent query expansion"""
        queries = [query]
        
        # Add code-specific variations
        if 'how' in query.lower():
            queries.append(query.replace('how', 'implementation of'))
        if 'what' in query.lower():
            queries.append(query.replace('what', 'definition of'))
        if 'error' in query.lower():
            queries.append(query + ' exception handling')
        
        # Add technical synonyms
        synonyms = {
            'function': 'method',
            'class': 'object',
            'bug': 'error',
            'create': 'implement'
        }
        
        for old, new in synonyms.items():
            if old in query.lower():
                queries.append(query.lower().replace(old, new))
        
        return queries[:3]
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Deduplicate and score chunks"""
        seen = {}
        for chunk in chunks:
            chunk_id = chunk['metadata'].get('chunk_id', chunk['content'][:100])
            if chunk_id not in seen:
                seen[chunk_id] = chunk
            else:
                # Boost score if seen multiple times
                seen[chunk_id]['score'] = seen[chunk_id].get('score', 0) + chunk.get('score', 0)
        
        return sorted(seen.values(), key=lambda x: x.get('score', 0), reverse=True)
    
    def _generate_answer(self, query: str, chunks: List[Dict]) -> str:
        """Generate answer using LLM"""
        
        system_prompt = """You are an expert code assistant with deep knowledge of software engineering.

Analyze the provided code chunks and answer the user's question with:
- Clear, accurate explanations
- Specific code examples when relevant
- Best practices and patterns
- Implementation guidance

Keep responses focused and well-structured."""
        
        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk['metadata']
            score = chunk.get('rerank_score', chunk.get('score', 0))
            
            context_parts.append(f"""
[CHUNK {i}] Score: {score:.3f}
File: {meta.get('filename', 'unknown')} ({meta.get('language', 'unknown')})
Repository: {meta.get('repo_name', 'unknown')}
Type: {meta.get('chunk_type', 'code')}
Lines: {meta.get('start_line', '?')}-{meta.get('end_line', '?')}

```{meta.get('language', '')}
{chunk['content']}
```
""")
        
        context = '\n'.join(context_parts)
        
        prompt = f"""{system_prompt}

CONTEXT:
{context}

QUESTION: {query}

Provide a comprehensive answer based on the code context:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Failed to generate answer: {str(e)}"
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            data = self.collection.get()
            
            if not data['metadatas']:
                return {'total_chunks': 0, 'languages': {}, 'repositories': {}}
            
            languages = Counter()
            repos = Counter()
            chunk_types = Counter()
            
            for meta in data['metadatas']:
                languages[meta.get('language', 'unknown')] += 1
                repos[meta.get('repo_name', 'unknown')] += 1
                chunk_types[meta.get('chunk_type', 'block')] += 1
            
            return {
                'total_chunks': len(data['ids']),
                'languages': dict(languages),
                'repositories': dict(repos),
                'chunk_types': dict(chunk_types)
            }
        except Exception as e:
            logger.error(f"Stats failed: {e}")
            return {'error': str(e)}
