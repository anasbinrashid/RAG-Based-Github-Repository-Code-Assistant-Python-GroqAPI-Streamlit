#!/usr/bin/env python3

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
from groq import Groq
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class CodeAgent:
    def __init__(self, db_path: str = "data/chromadb", model: str = "llama3-70b-8192"):
        # Initialize Groq
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable required")
        
        self.client = Groq(api_key=self.groq_api_key)
        self.model = model
        
        # Initialize ChromaDB
        self.db_path = Path(db_path)
        chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="jinaai/jina-embeddings-v2-base-code"
        )
        
        try:
            self.collection = chroma_client.get_collection("code_chunks", embedding_function=embedding_fn)
        except:
            raise ValueError("No code collection found. Run processor first.")
        
        logger.info(f"Initialized CodeAgent with {self.model}")
    
    def _expand_query(self, query: str) -> List[str]:
        """Simple query expansion"""
        # Basic synonyms
        synonyms = {
            'function': ['method', 'procedure'],
            'class': ['object', 'component'],
            'error': ['exception', 'bug'],
            'implement': ['create', 'build'],
            'find': ['search', 'locate'],
            'explain': ['describe', 'clarify']
        }
        
        queries = [query]
        query_lower = query.lower()
        
        # Add synonym variations (limit to 2-3 expansions)
        for term, alternatives in synonyms.items():
            if term in query_lower:
                for alt in alternatives[:2]:  # Max 2 alternatives per term
                    new_query = query_lower.replace(term, alt)
                    if new_query not in queries:
                        queries.append(new_query)
        
        # Add simple variations
        if 'how to' in query_lower:
            queries.append(query_lower.replace('how to', 'implementing'))
        if 'what is' in query_lower:
            queries.append(query_lower.replace('what is', 'definition of'))
        
        return queries[:4]  # Max 4 total queries
    
    def _search_chunks(self, query: str, n_results: int = 5, language: Optional[str] = None) -> List[Dict]:
        """Search code chunks with query expansion"""
        try:
            # Get expanded queries
            queries = self._expand_query(query)
            
            # Search with each query and combine results
            all_chunks = {}
            
            for i, q in enumerate(queries):
                weight = 1.0 if i == 0 else 0.6  # Original query gets full weight
                where_clause = {'language': language} if language else None
                
                results = self.collection.query(
                    query_texts=[q],
                    n_results=n_results,
                    where=where_clause
                )
                
                if results['documents'] and results['documents'][0]:
                    for doc, metadata, distance in zip(
                        results['documents'][0], 
                        results['metadatas'][0], 
                        results['distances'][0]
                    ):
                        chunk_id = f"{metadata['filename']}:{metadata.get('start_line', 0)}"
                        relevance = (1 - distance) * weight if distance else 0
                        
                        if chunk_id in all_chunks:
                            # Boost chunks found multiple times
                            all_chunks[chunk_id]['relevance_score'] += relevance * 0.3
                        else:
                            all_chunks[chunk_id] = {
                                'filename': metadata.get('filename', 'Unknown'),
                                'repo_name': metadata.get('repo_name', 'Unknown'),
                                'language': metadata.get('language', 'Unknown'),
                                'lines': f"{metadata.get('start_line', 0)}-{metadata.get('end_line', 0)}",
                                'content': doc,
                                'relevance_score': relevance,
                                'metadata': metadata
                            }
            
            # Sort by relevance and return top results
            sorted_chunks = sorted(all_chunks.values(), key=lambda x: x['relevance_score'], reverse=True)
            return sorted_chunks[:n_results]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _analyze_query(self, query: str) -> Dict:
        """Simple query analysis"""
        query_lower = query.lower()
        
        # Detect intent
        if any(word in query_lower for word in ['explain', 'what', 'how does', 'describe']):
            intent = 'explanation'
        elif any(word in query_lower for word in ['find', 'search', 'show', 'locate']):
            intent = 'search'
        elif any(word in query_lower for word in ['implement', 'create', 'build', 'make']):
            intent = 'implementation'
        elif any(word in query_lower for word in ['debug', 'fix', 'error', 'bug']):
            intent = 'debugging'
        else:
            intent = 'general'
        
        # Detect language
        language = None
        lang_patterns = {
            'python': r'\b(python|py|def |class |import |\.py)\b',
            'javascript': r'\b(javascript|js|function|const|let|var|\.js)\b',
            'java': r'\b(java|class|public|private|static|\.java)\b',
            'cpp': r'\b(c\+\+|cpp|#include|namespace|\.cpp)\b'
        }
        
        for lang, pattern in lang_patterns.items():
            if re.search(pattern, query_lower):
                language = lang
                break
        
        return {
            'intent': intent,
            'language': language,
            'n_results': 8 if intent == 'implementation' else 5
        }
    
    def _build_prompt(self, query: str, chunks: List[Dict], analysis: Dict) -> str:
        """Build context prompt for LLM"""
        
        system_prompt = """You are an expert code assistant. Analyze the provided code chunks and answer the user's question accurately and concisely.

Guidelines:
- Give direct, well-structured answers
- Reference specific code examples when helpful
- Keep responses comprehensive but not overwhelming  
- Use clear formatting with headers and code blocks
- If information is incomplete, state limitations clearly"""
        
        # Build context from chunks
        context_sections = []
        for i, chunk in enumerate(chunks, 1):
            context_sections.append(f"""
CODE CHUNK {i}:
File: {chunk['filename']} ({chunk['language']})
Repository: {chunk['repo_name']}
Lines: {chunk['lines']}
Relevance: {chunk['relevance_score']:.3f}

```{chunk['language']}
{chunk['content']}
```
""")
        
        # Combine everything
        prompt = f"""{system_prompt}

CONTEXT INFORMATION:
{''.join(context_sections)}

USER QUERY: {query}

Please provide a helpful response based on the code context above:"""
        
        return prompt
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"Error processing request: {str(e)}"
    
    def query(self, query: str, language: Optional[str] = None) -> Dict:
        """Process user query and return response"""
        start_time = datetime.now()
        
        try:
            # Analyze query
            analysis = self._analyze_query(query)
            if not language:
                language = analysis['language']
            
            logger.info(f"Query intent: {analysis['intent']}, language: {language}")
            
            # Search for relevant chunks
            chunks = self._search_chunks(query, analysis['n_results'], language)
            
            if not chunks:
                return {
                    'success': False,
                    'answer': "No relevant code found. Make sure repositories are processed.",
                    'sources': [],
                    'query': query
                }
            
            logger.info(f"Found {len(chunks)} relevant chunks")
            
            # Build prompt and get response
            prompt = self._build_prompt(query, chunks, analysis)
            answer = self._call_groq(prompt)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare sources
            sources = []
            for chunk in chunks:
                sources.append({
                    'filename': chunk['filename'],
                    'repository': chunk['repo_name'],
                    'language': chunk['language'],
                    'lines': chunk['lines'],
                    'relevance_score': round(chunk['relevance_score'], 3)
                })
            
            return {
                'success': True,
                'query': query,
                'answer': answer,
                'sources': sources,
                'response_time': response_time,
                'model_used': self.model,
                'chunks_analyzed': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {
                'success': False,
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'query': query
            }
    
    def search_code(self, query: str, limit: int = 5, language: Optional[str] = None) -> Dict:
        """Simple code search without LLM processing"""
        try:
            chunks = self._search_chunks(query, limit, language)
            
            results = []
            for chunk in chunks:
                results.append({
                    'filename': chunk['filename'],
                    'repository': chunk['repo_name'],
                    'language': chunk['language'],
                    'lines': chunk['lines'],
                    'relevance_score': round(chunk['relevance_score'], 3),
                    'preview': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                })
            
            return {
                'success': True,
                'results': results,
                'total_found': len(results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            all_data = self.collection.get()
            
            if not all_data['metadatas']:
                return {'success': True, 'total_chunks': 0, 'languages': {}, 'repositories': {}}
            
            languages = {}
            repos = {}
            
            for metadata in all_data['metadatas']:
                lang = metadata.get('language', 'unknown')
                repo = metadata.get('repo_name', 'unknown')
                
                languages[lang] = languages.get(lang, 0) + 1
                repos[repo] = repos.get(repo, 0) + 1
            
            return {
                'success': True,
                'total_chunks': len(all_data['ids']),
                'languages': languages,
                'repositories': repos
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

class InteractiveAgent:
    """Interactive interface for the agent"""
    
    def __init__(self, db_path: str = "data/chromadb", model: str = "llama3-70b-8192"):
        self.agent = CodeAgent(db_path, model)
        self.history = []
    
    def start(self):
        """Start interactive session"""
        print("Code Assistant (Powered by Groq)")
        print("="*50)
        print("Ask questions about your codebase. Type 'help' for commands or 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("Query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                
                if user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                # Process query
                print("\nProcessing...")
                response = self.agent.query(user_input)
                
                # Display response
                self._display_response(response)
                
                # Add to history
                self.history.append({
                    'query': user_input,
                    'success': response['success'],
                    'sources': len(response.get('sources', [])),
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_help(self):
        """Show help"""
        print("""
Commands:
  help     - Show this help
  stats    - Show database statistics  
  history  - Show query history
  quit     - Exit

Examples:
  "Explain how authentication works"
  "Find Python error handling examples"
  "How to implement REST API?"
  "Show me database connection code"
""")
    
    def _show_stats(self):
        """Show statistics"""
        stats = self.agent.get_stats()
        if stats['success']:
            print(f"\nDatabase Statistics:")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Languages: {len(stats['languages'])}")
            print(f"  Repositories: {len(stats['repositories'])}")
            
            # Top languages
            if stats['languages']:
                print(f"\nTop Languages:")
                for lang, count in sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {lang}: {count}")
        else:
            print(f"Error getting stats: {stats.get('error')}")
    
    def _show_history(self):
        """Show query history"""
        if not self.history:
            print("No queries yet.")
            return
        
        print(f"\nQuery History ({len(self.history)} queries):")
        for i, item in enumerate(self.history[-5:], 1):  # Show last 5
            status = "✓" if item['success'] else "✗"
            print(f"{i}. [{item['timestamp']}] {status} {item['query'][:50]}...")
            print(f"   Sources: {item['sources']}")
    
    def _display_response(self, response: Dict):
        """Display response"""
        print("\n" + "="*60)
        if response['success']:
            print("Answer:")
            print(response['answer'])
            
            print(f"\nDetails:")
            print(f"  Response time: {response.get('response_time', 0):.2f}s")
            print(f"  Sources analyzed: {len(response.get('sources', []))}")
            print(f"  Model: {response.get('model_used', 'unknown')}")
            
            if response.get('sources'):
                print(f"\nSources:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"  {i}. {source['filename']} ({source['language']})")
                    print(f"     {source['repository']} - Lines {source['lines']}")
                    print(f"     Relevance: {source['relevance_score']}")
        else:
            print(f"Error: {response['answer']}")
        
        print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Streamlined Code Agent")
    parser.add_argument("--query", help="Single query to process")
    parser.add_argument("--search", help="Search code without LLM processing")
    parser.add_argument("--language", help="Filter by programming language")
    parser.add_argument("--model", default="llama3-70b-8192", help="Groq model to use")
    parser.add_argument("--db-path", default="data/chromadb", help="Database path")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--limit", type=int, default=5, help="Result limit for search")
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY required")
        return
    
    try:
        if args.interactive or not (args.query or args.search or args.stats):
            agent = InteractiveAgent(args.db_path, args.model)
            agent.start()
        else:
            agent = CodeAgent(args.db_path, args.model)
            
            if args.stats:
                stats = agent.get_stats()
                if stats['success']:
                    print(f"Total chunks: {stats['total_chunks']}")
                    print(f"Languages: {stats['languages']}")
                    print(f"Repositories: {stats['repositories']}")
                else:
                    print(f"Error: {stats['error']}")
            
            elif args.search:
                result = agent.search_code(args.search, args.limit, args.language)
                if result['success']:
                    print(f"Found {result['total_found']} results:")
                    for i, res in enumerate(result['results'], 1):
                        print(f"{i}. {res['filename']} ({res['language']})")
                        print(f"   {res['repository']} - Lines {res['lines']}")
                        print(f"   Relevance: {res['relevance_score']}")
                        print(f"   {res['preview']}\n")
                else:
                    print(f"Error: {result['error']}")
            
            elif args.query:
                response = agent.query(args.query, args.language)
                if response['success']:
                    print(f"Answer:\n{response['answer']}")
                    print(f"\nSources: {len(response['sources'])}")
                    print(f"Time: {response['response_time']:.2f}s")
                else:
                    print(f"Error: {response['answer']}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()