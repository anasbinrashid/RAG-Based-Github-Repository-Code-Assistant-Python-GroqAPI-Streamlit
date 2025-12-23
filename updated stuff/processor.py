#!/usr/bin/env python3
"""
Repository Processing Pipeline
Handles cloning, parsing, and indexing with progress tracking
"""

import os
import git
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from core import EnhancedRAGEngine, CodeChunk

logger = logging.getLogger(__name__)

class RepositoryProcessor:
    """Process repositories with parallel execution"""
    
    SUPPORTED_EXTENSIONS = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
        '.jsx': 'javascript', '.tsx': 'typescript', '.java': 'java',
        '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.hpp': 'cpp',
        '.cs': 'csharp', '.go': 'go', '.rs': 'rust', '.php': 'php',
        '.rb': 'ruby', '.html': 'html', '.css': 'css', '.sql': 'sql',
        '.sh': 'bash', '.kt': 'kotlin', '.swift': 'swift'
    }
    
    SKIP_DIRS = {
        '.git', 'node_modules', '__pycache__', '.pytest_cache',
        'venv', 'env', '.venv', 'build', 'dist', 'target',
        '.idea', '.vscode', 'coverage', '.next', 'vendor',
        '.bundle', 'bower_components', '.cache'
    }
    
    def __init__(self, engine: EnhancedRAGEngine, base_dir: str = "data/repos"):
        self.engine = engine
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def process_repository(self, repo_url: str, repo_name: Optional[str] = None, 
                          max_workers: int = 4) -> Dict:
        """Process entire repository with parallel file processing"""
        
        try:
            # Clone repository
            repo_path = self._clone_repo(repo_url, repo_name)
            repo_name = repo_path.name
            
            logger.info(f"Processing repository: {repo_name}")
            
            # Find all processable files
            files = self._find_files(repo_path)
            
            if not files:
                return {
                    'success': False,
                    'error': 'No processable files found',
                    'repo_name': repo_name
                }
            
            logger.info(f"Found {len(files)} files to process")
            
            # Clear existing data for this repo
            self._clear_repo_data(repo_name)
            
            # Process files in parallel
            all_chunks = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_file, f, repo_path, repo_name): f 
                    for f in files
                }
                
                with tqdm(total=len(files), desc="Processing files") as pbar:
                    for future in as_completed(futures):
                        try:
                            chunks = future.result()
                            all_chunks.extend(chunks)
                        except Exception as e:
                            logger.error(f"File processing error: {e}")
                        pbar.update(1)
            
            # Batch insert chunks
            if all_chunks:
                self._insert_chunks(all_chunks)
            
            # Rebuild BM25 index
            self.engine.retriever._build_bm25_index()
            
            logger.info(f"Completed: {repo_name} - {len(files)} files, {len(all_chunks)} chunks")
            
            return {
                'success': True,
                'repo_name': repo_name,
                'files_processed': len(files),
                'chunks_created': len(all_chunks),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Repository processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'repo_name': repo_name or 'unknown'
            }
    
    def _clone_repo(self, repo_url: str, name: Optional[str] = None) -> Path:
        """Clone or update repository"""
        if not name:
            name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        
        repo_path = self.base_dir / name
        
        if repo_path.exists():
            logger.info(f"Removing existing repo: {name}")
            shutil.rmtree(repo_path)
        
        logger.info(f"Cloning {repo_url}")
        git.Repo.clone_from(repo_url, repo_path, depth=1)
        
        return repo_path
    
    def _find_files(self, repo_path: Path) -> List[Path]:
        """Find all processable files"""
        files = []
        
        for file_path in repo_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Skip if in excluded directory
            if any(part in self.SKIP_DIRS for part in file_path.parts):
                continue
            
            # Check extension
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            
            # Skip large files (>1MB)
            try:
                if file_path.stat().st_size > 1024 * 1024:
                    continue
            except:
                continue
            
            files.append(file_path)
        
        return files
    
    def _process_file(self, file_path: Path, repo_path: Path, repo_name: str) -> List[CodeChunk]:
        """Process single file into chunks"""
        try:
            # Read file
            content = self._read_file(file_path)
            if not content:
                return []
            
            # Get language
            language = self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), 'unknown')
            
            # Create chunks
            relative_path = file_path.relative_to(repo_path)
            chunks = self.engine.chunker.chunk_code(content, relative_path, repo_name, language)
            
            return chunks
            
        except Exception as e:
            logger.debug(f"Failed to process {file_path}: {e}")
            return []
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file with encoding fallback"""
        for encoding in ['utf-8', 'latin-1', 'ascii', 'cp1252']:
            try:
                return file_path.read_text(encoding=encoding)
            except:
                continue
        return None
    
    def _clear_repo_data(self, repo_name: str):
        """Remove existing data for repository"""
        try:
            existing = self.engine.collection.get(where={'repo_name': repo_name})
            if existing['ids']:
                self.engine.collection.delete(ids=existing['ids'])
                logger.info(f"Cleared {len(existing['ids'])} existing chunks for {repo_name}")
        except Exception as e:
            logger.warning(f"Failed to clear existing data: {e}")
    
    def _insert_chunks(self, chunks: List[CodeChunk], batch_size: int = 100):
        """Insert chunks into database in batches"""
        logger.info(f"Inserting {len(chunks)} chunks")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = [c.id for c in batch]
            documents = [c.content for c in batch]
            metadatas = []
            
            for chunk in batch:
                meta = {
                    'chunk_id': chunk.id,
                    'filename': chunk.filename,
                    'filepath': chunk.filepath,
                    'repo_name': chunk.repo_name,
                    'language': chunk.language,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'chunk_type': chunk.chunk_type,
                    'complexity': chunk.complexity,
                    'token_count': chunk.token_count
                }
                
                if chunk.symbols:
                    meta['symbols'] = ','.join(chunk.symbols)
                if chunk.docstring:
                    meta['has_docstring'] = True
                
                metadatas.append(meta)
            
            self.engine.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )


def main():
    """CLI for repository processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process code repositories")
    parser.add_argument("repo_url", help="Repository URL")
    parser.add_argument("--name", help="Repository name")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--db-path", default="data/chromadb_v2", help="Database path")
    
    args = parser.parse_args()
    
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY required")
        return
    
    try:
        engine = EnhancedRAGEngine(db_path=args.db_path)
        processor = RepositoryProcessor(engine)
        
        result = processor.process_repository(
            args.repo_url,
            args.name,
            max_workers=args.workers
        )
        
        if result['success']:
            print(f"✓ Success: {result['repo_name']}")
            print(f"  Files: {result['files_processed']}")
            print(f"  Chunks: {result['chunks_created']}")
        else:
            print(f"✗ Error: {result['error']}")
            
    except Exception as e:
        print(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
