#!/usr/bin/env python3

import git
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import re
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class GitHubProcessor:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # File extensions to process
        self.extensions = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript', 
            '.jsx': 'javascript', '.tsx': 'typescript', '.java': 'java', 
            '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.hpp': 'cpp', 
            '.cs': 'csharp', '.go': 'go', '.rs': 'rust', '.php': 'php', 
            '.rb': 'ruby', '.html': 'html', '.css': 'css', '.sql': 'sql', 
            '.sh': 'bash', '.json': 'json', '.md': 'markdown', 
            '.yaml': 'yaml', '.yml': 'yaml', '.kt': 'kotlin', 
            '.swift': 'swift', '.scala': 'scala'
        }
        
        # Directories to skip
        self.skip_dirs = {
            '.git', 'node_modules', '__pycache__', '.pytest_cache',
            'venv', 'env', '.venv', 'build', 'dist', 'target',
            '.idea', '.vscode', 'coverage', '.next', 'vendor'
        }
        
        # Initialize ChromaDB
        db_path = self.base_dir / "chromadb"
        db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(db_path))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="jinaai/jina-embeddings-v2-base-code"
        )
        
        # Handle existing collections
        try:
            self.collection = self.client.get_collection("code_chunks", embedding_function=embedding_fn)
        except ValueError:
            # Delete and recreate if embedding function conflicts
            try:
                self.client.delete_collection("code_chunks")
            except:
                pass
            self.collection = self.client.create_collection("code_chunks", embedding_function=embedding_fn)
        except:
            self.collection = self.client.create_collection("code_chunks", embedding_function=embedding_fn)
    
    def clone_repo(self, repo_url: str, name: Optional[str] = None) -> Path:
        """Clone repository"""
        if not name:
            name = repo_url.split('/')[-1].replace('.git', '')
        
        repo_path = self.base_dir / "repos" / name
        
        if repo_path.exists():
            import shutil
            shutil.rmtree(repo_path)
        
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        git.Repo.clone_from(repo_url, repo_path, depth=1)
        logger.info(f"Cloned {repo_url}")
        return repo_path
    
    def should_process(self, file_path: Path) -> bool:
        """Check if file should be processed"""
        # Skip directories
        if any(part in self.skip_dirs for part in file_path.parts):
            return False
        
        # Check extension
        if file_path.suffix.lower() not in self.extensions:
            return False
        
        # Skip large files
        try:
            return file_path.stat().st_size <= 1024 * 1024  # 1MB limit
        except:
            return False
    
    def read_file(self, file_path: Path) -> Optional[str]:
        """Read file with encoding fallback"""
        for encoding in ['utf-8', 'latin-1', 'ascii']:
            try:
                return file_path.read_text(encoding=encoding)
            except:
                continue
        return None
    
    def chunk_content(self, content: str, file_path: Path, repo_name: str, chunk_size: int = 75) -> List[Dict]:
        """Split content into overlapping chunks"""
        lines = content.split('\n')
        language = self.extensions.get(file_path.suffix.lower(), 'unknown')
        chunks = []
        
        if len(lines) <= chunk_size:
            chunk_id = hashlib.md5(f"{repo_name}:{file_path}:1-{len(lines)}".encode()).hexdigest()[:12]
            chunks.append({
                'id': chunk_id,
                'content': content,
                'metadata': {
                    'filename': file_path.name,
                    'file_path': str(file_path),
                    'language': language,
                    'start_line': 1,
                    'end_line': len(lines),
                    'repo_name': repo_name,
                    'size_lines': len(lines)
                }
            })
        else:
            overlap = 10
            start = 0
            
            while start < len(lines):
                end = min(start + chunk_size, len(lines))
                chunk_lines = lines[start:end]
                chunk_content = '\n'.join(chunk_lines)
                
                chunk_id = hashlib.md5(f"{repo_name}:{file_path}:{start+1}-{end}".encode()).hexdigest()[:12]
                chunks.append({
                    'id': chunk_id,
                    'content': chunk_content,
                    'metadata': {
                        'filename': file_path.name,
                        'file_path': str(file_path),
                        'language': language,
                        'start_line': start + 1,
                        'end_line': end,
                        'repo_name': repo_name,
                        'size_lines': len(chunk_lines)
                    }
                })
                
                start = end - overlap
                if end >= len(lines):
                    break
        
        return chunks
    
    def extract_repo_info(self, repo_path: Path, repo_name: str, repo_url: str) -> Dict:
        """Extract basic repository metadata"""
        files = [f for f in repo_path.rglob('*') if f.is_file() and self.should_process(f)]
        
        # Count languages
        languages = Counter()
        for file_path in files:
            lang = self.extensions.get(file_path.suffix.lower(), 'unknown')
            languages[lang] += 1
        
        # Get README
        readme = ""
        for readme_file in ['README.md', 'README.rst', 'README.txt', 'README']:
            readme_path = repo_path / readme_file
            if readme_path.exists():
                readme = self.read_file(readme_path) or ""
                break
        
        # Extract dependencies (simple version)
        deps = []
        
        # Python
        req_file = repo_path / 'requirements.txt'
        if req_file.exists():
            content = self.read_file(req_file)
            if content:
                deps.extend(re.findall(r'^([a-zA-Z0-9-_]+)', content, re.MULTILINE))
        
        # JavaScript
        package_json = repo_path / 'package.json'
        if package_json.exists():
            content = self.read_file(package_json)
            if content:
                try:
                    data = json.loads(content)
                    deps.extend(data.get('dependencies', {}).keys())
                except:
                    pass
        
        return {
            'repo_name': repo_name,
            'repo_url': repo_url,
            'primary_language': languages.most_common(1)[0][0] if languages else 'unknown',
            'languages': dict(languages),
            'total_files': len(files),
            'dependencies': deps,
            'readme': readme[:500],
            'created_at': datetime.now().isoformat()
        }
    
    def process_repository(self, repo_url: str, repo_name: Optional[str] = None) -> Dict:
        """Process entire repository"""
        try:
            # Clone
            repo_path = self.clone_repo(repo_url, repo_name)
            repo_name = repo_path.name
            
            # Get repo info
            repo_info = self.extract_repo_info(repo_path, repo_name, repo_url)
            
            # Find files
            files = [f for f in repo_path.rglob('*') if f.is_file() and self.should_process(f)]
            
            # Process files and create chunks
            all_chunks = []
            for file_path in files:
                content = self.read_file(file_path)
                if content:
                    relative_path = file_path.relative_to(repo_path)
                    chunks = self.chunk_content(content, relative_path, repo_name)
                    all_chunks.extend(chunks)
            
            # Delete existing data for this repo
            try:
                existing = self.collection.get(where={'repo_name': repo_name})
                if existing['ids']:
                    self.collection.delete(ids=existing['ids'])
            except:
                pass
            
            # Add to ChromaDB in batches
            if all_chunks:
                batch_size = 100
                for i in range(0, len(all_chunks), batch_size):
                    batch = all_chunks[i:i + batch_size]
                    
                    # Enhanced documents with repo context
                    documents = []
                    metadatas = []
                    ids = []
                    
                    for chunk in batch:
                        # Add repo context to content
                        enhanced_content = f"""Repository: {repo_name}
Primary Language: {repo_info['primary_language']}
Dependencies: {', '.join(repo_info['dependencies'][:10])}
File: {chunk['metadata']['filename']}

{chunk['content']}"""
                        
                        documents.append(enhanced_content)
                        
                        # Enhanced metadata
                        metadata = chunk['metadata'].copy()
                        metadata.update({
                            'repo_primary_language': repo_info['primary_language'],
                            'repo_dependencies': json.dumps(repo_info['dependencies'][:20]),
                            'repo_total_files': repo_info['total_files']
                        })
                        metadatas.append(metadata)
                        ids.append(chunk['id'])
                    
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
            
            logger.info(f"Processed {repo_name}: {len(files)} files, {len(all_chunks)} chunks")
            
            return {
                'success': True,
                'repo_name': repo_name,
                'files_processed': len(files),
                'chunks_created': len(all_chunks),
                'repo_info': repo_info
            }
            
        except Exception as e:
            logger.error(f"Error processing repository: {e}")
            return {'success': False, 'error': str(e)}
    
    def search(self, query: str, limit: int = 5, language: Optional[str] = None) -> Dict:
        """Search code chunks"""
        try:
            where_clause = {'language': language} if language else None
            
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause
            )
            
            if not results['documents']:
                return {'success': True, 'results': []}
            
            formatted_results = []
            for doc, metadata, distance in zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            ):
                formatted_results.append({
                    'filename': metadata['filename'],
                    'language': metadata['language'],
                    'repo_name': metadata['repo_name'],
                    'lines': f"{metadata['start_line']}-{metadata['end_line']}",
                    'relevance_score': 1 - distance,
                    'preview': doc.split('\n\n')[-1][:200] + "..." if len(doc.split('\n\n')[-1]) > 200 else doc.split('\n\n')[-1]
                })
            
            return {
                'success': True,
                'results': formatted_results,
                'total_found': len(formatted_results)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            all_data = self.collection.get()
            
            if not all_data['metadatas']:
                return {'success': True, 'total_chunks': 0, 'languages': {}, 'repositories': {}}
            
            languages = Counter()
            repos = Counter()
            
            for metadata in all_data['metadatas']:
                languages[metadata.get('language', 'unknown')] += 1
                repos[metadata.get('repo_name', 'unknown')] += 1
            
            return {
                'success': True,
                'total_chunks': len(all_data['ids']),
                'languages': dict(languages),
                'repositories': dict(repos)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Streamlined GitHub Code Processor")
    parser.add_argument("command", choices=["process", "search", "stats"])
    parser.add_argument("--repo", help="GitHub repository URL")
    parser.add_argument("--name", help="Repository name")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--language", help="Filter by language")
    parser.add_argument("--limit", type=int, default=5, help="Result limit")
    parser.add_argument("--base-dir", default="data", help="Base directory")
    
    args = parser.parse_args()
    processor = GitHubProcessor(args.base_dir)
    
    if args.command == "process":
        if not args.repo:
            print("Error: --repo required")
            return
        
        result = processor.process_repository(args.repo, args.name)
        if result['success']:
            print(f"Success: {result['repo_name']} - {result['files_processed']} files, {result['chunks_created']} chunks")
        else:
            print(f"Error: {result['error']}")
    
    elif args.command == "search":
        if not args.query:
            print("Error: --query required")
            return
        
        result = processor.search(args.query, args.limit, args.language)
        if result['success']:
            for i, res in enumerate(result['results'], 1):
                print(f"{i}. {res['filename']} ({res['language']}) - {res['repo_name']}")
                print(f"   Lines {res['lines']}, Score: {res['relevance_score']:.3f}")
                print(f"   {res['preview']}\n")
        else:
            print(f"Error: {result['error']}")
    
    elif args.command == "stats":
        result = processor.get_stats()
        if result['success']:
            print(f"Total chunks: {result['total_chunks']}")
            print(f"Languages: {result['languages']}")
            print(f"Repositories: {result['repositories']}")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()