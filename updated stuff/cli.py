#!/usr/bin/env python3
"""
Interactive CLI for Code RAG System
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from core import EnhancedRAGEngine
from processor import RepositoryProcessor

logging.basicConfig(level=logging.WARNING)

class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class InteractiveCLI:
    """Enhanced interactive CLI"""
    
    def __init__(self):
        self.engine: Optional[EnhancedRAGEngine] = None
        self.processor: Optional[RepositoryProcessor] = None
        self.history = []
        
    def start(self):
        """Start interactive session"""
        self.print_header()
        
        # Check API key
        if not os.getenv("GROQ_API_KEY"):
            print(f"{Colors.RED}Error: GROQ_API_KEY environment variable required!{Colors.END}")
            print("Set it in .env file or export GROQ_API_KEY=your_key")
            return
        
        # Initialize
        try:
            print(f"\n{Colors.CYAN}Initializing RAG engine...{Colors.END}")
            self.engine = EnhancedRAGEngine()
            self.processor = RepositoryProcessor(self.engine)
            print(f"{Colors.GREEN}✓ Ready!{Colors.END}\n")
        except Exception as e:
            print(f"{Colors.RED}Initialization failed: {e}{Colors.END}")
            return
        
        # Show stats
        self.cmd_stats()
        
        # Main loop
        while True:
            try:
                user_input = input(f"\n{Colors.BOLD}> {Colors.END}").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(f"\n{Colors.CYAN}Goodbye!{Colors.END}")
                    break
                
                elif user_input.lower() == 'help':
                    self.cmd_help()
                
                elif user_input.lower() == 'stats':
                    self.cmd_stats()
                
                elif user_input.lower() == 'history':
                    self.cmd_history()
                
                elif user_input.lower() == 'clear':
                    self.cmd_clear()
                
                elif user_input.lower().startswith('process '):
                    url = user_input[8:].strip()
                    self.cmd_process(url)
                
                elif user_input.lower().startswith('search '):
                    query = user_input[7:].strip()
                    self.cmd_search(query)
                
                else:
                    # Treat as query
                    self.cmd_query(user_input)
                
            except KeyboardInterrupt:
                print(f"\n\n{Colors.CYAN}Goodbye!{Colors.END}")
                break
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.END}")
    
    def print_header(self):
        """Print welcome header"""
        print(f"""
{Colors.BOLD}{Colors.BLUE}╔═══════════════════════════════════════════════════════════╗
║                  Code RAG Assistant                       ║
║               Advanced Semantic Code Search               ║
╚═══════════════════════════════════════════════════════════╝{Colors.END}

Type 'help' for commands or ask a question directly.
""")
    
    def cmd_help(self):
        """Show help"""
        print(f"""
{Colors.BOLD}Commands:{Colors.END}
  {Colors.CYAN}help{Colors.END}              Show this help message
  {Colors.CYAN}stats{Colors.END}             Show database statistics
  {Colors.CYAN}history{Colors.END}           Show query history
  {Colors.CYAN}clear{Colors.END}             Clear screen and history
  {Colors.CYAN}process <url>{Colors.END}     Process a repository
  {Colors.CYAN}search <query>{Colors.END}    Quick code search (no LLM)
  {Colors.CYAN}quit{Colors.END}              Exit

{Colors.BOLD}Query Examples:{Colors.END}
  How is authentication implemented?
  Show me error handling patterns
  Find database connection code
  Explain the API structure
""")
    
    def cmd_stats(self):
        """Show statistics"""
        stats = self.engine.get_stats()
        
        if 'error' in stats:
            print(f"{Colors.RED}Error: {stats['error']}{Colors.END}")
            return
        
        print(f"\n{Colors.BOLD}Database Statistics:{Colors.END}")
        print(f"  Total Chunks: {Colors.GREEN}{stats.get('total_chunks', 0):,}{Colors.END}")
        
        # Languages
        languages = stats.get('languages', {})
        if languages:
            print(f"\n  {Colors.BOLD}Languages:{Colors.END}")
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]:
                bar = '█' * min(int(count / max(languages.values()) * 30), 30)
                print(f"    {lang:12s} {bar} {count:,}")
        
        # Repositories
        repos = stats.get('repositories', {})
        if repos:
            print(f"\n  {Colors.BOLD}Repositories:{Colors.END}")
            for repo, count in list(repos.items())[:5]:
                print(f"    • {repo}: {count:,} chunks")
        
        # Chunk types
        chunk_types = stats.get('chunk_types', {})
        if chunk_types:
            print(f"\n  {Colors.BOLD}Chunk Types:{Colors.END}")
            for ctype, count in sorted(chunk_types.items(), key=lambda x: x[1], reverse=True):
                print(f"    • {ctype}: {count:,}")
    
    def cmd_history(self):
        """Show history"""
        if not self.history:
            print(f"{Colors.YELLOW}No queries yet.{Colors.END}")
            return
        
        print(f"\n{Colors.BOLD}Query History ({len(self.history)} queries):{Colors.END}")
        for i, item in enumerate(self.history[-10:], 1):
            status = f"{Colors.GREEN}✓{Colors.END}" if item['success'] else f"{Colors.RED}✗{Colors.END}"
            print(f"{i}. [{item['timestamp']}] {status} {item['query'][:60]}...")
            print(f"   Sources: {item['sources']}, Time: {item['time']:.2f}s")
    
    def cmd_clear(self):
        """Clear screen and history"""
        os.system('clear' if os.name != 'nt' else 'cls')
        self.history.clear()
        self.print_header()
        print(f"{Colors.GREEN}History cleared.{Colors.END}")
    
    def cmd_process(self, url: str):
        """Process repository"""
        if not url:
            print(f"{Colors.RED}Usage: process <repository_url>{Colors.END}")
            return
        
        print(f"\n{Colors.CYAN}Processing {url}...{Colors.END}")
        
        try:
            result = self.processor.process_repository(url, max_workers=4)
            
            if result['success']:
                print(f"{Colors.GREEN}✓ Success!{Colors.END}")
                print(f"  Repository: {result['repo_name']}")
                print(f"  Files: {result['files_processed']}")
                print(f"  Chunks: {result['chunks_created']}")
            else:
                print(f"{Colors.RED}✗ Failed: {result['error']}{Colors.END}")
        
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.END}")
    
    def cmd_search(self, query: str):
        """Quick search without LLM"""
        if not query:
            print(f"{Colors.RED}Usage: search <query>{Colors.END}")
            return
        
        print(f"\n{Colors.CYAN}Searching...{Colors.END}")
        
        try:
            chunks = self.engine.retriever.search(query, n_results=5)
            
            if not chunks:
                print(f"{Colors.YELLOW}No results found.{Colors.END}")
                return
            
            print(f"\n{Colors.BOLD}Found {len(chunks)} results:{Colors.END}\n")
            
            for i, chunk in enumerate(chunks, 1):
                meta = chunk['metadata']
                score = chunk.get('score', 0)
                
                print(f"{Colors.BOLD}{i}. {meta.get('filename', 'unknown')}{Colors.END} ({meta.get('language', 'unknown')})")
                print(f"   {meta.get('repo_name', 'unknown')} • Lines {meta.get('start_line', '?')}-{meta.get('end_line', '?')}")
                print(f"   Score: {Colors.GREEN}{score:.3f}{Colors.END}")
                print(f"   {chunk['content'][:150]}...")
                print()
        
        except Exception as e:
            print(f"{Colors.RED}Search failed: {e}{Colors.END}")
    
    def cmd_query(self, query: str):
        """Full query with LLM"""
        print(f"\n{Colors.CYAN}Processing query...{Colors.END}")
        
        try:
            start = datetime.now()
            response = self.engine.query(query, n_results=8)
            elapsed = (datetime.now() - start).total_seconds()
            
            if not response['success']:
                print(f"{Colors.RED}Query failed: {response['answer']}{Colors.END}")
                return
            
            # Display answer
            print(f"\n{Colors.BOLD}{Colors.GREEN}Answer:{Colors.END}")
            print(f"{response['answer']}\n")
            
            # Display metrics
            chunks = response.get('chunks', [])
            print(f"{Colors.BOLD}Details:{Colors.END}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Sources: {len(chunks)}")
            print(f"  Model: {response.get('model', 'unknown')}")
            
            # Display sources
            if chunks:
                print(f"\n{Colors.BOLD}Sources:{Colors.END}")
                for i, chunk in enumerate(chunks[:5], 1):
                    meta = chunk['metadata']
                    score = chunk.get('rerank_score', chunk.get('score', 0))
                    print(f"{i}. {meta.get('filename', 'unknown')} ({meta.get('language', 'unknown')}) - Score: {score:.3f}")
                    print(f"   {meta.get('repo_name', 'unknown')} • Lines {meta.get('start_line', '?')}-{meta.get('end_line', '?')}")
            
            # Add to history
            self.history.append({
                'query': query,
                'success': True,
                'sources': len(chunks),
                'time': elapsed,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.END}")
            self.history.append({
                'query': query,
                'success': False,
                'sources': 0,
                'time': 0,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Code RAG Assistant CLI")
    parser.add_argument("--query", help="Single query mode")
    parser.add_argument("--process", help="Process repository")
    parser.add_argument("--stats", action="store_true", help="Show stats and exit")
    
    args = parser.parse_args()
    
    cli = InteractiveCLI()
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print(f"{Colors.RED}Error: GROQ_API_KEY required{Colors.END}")
        return
    
    # Initialize
    try:
        print(f"{Colors.CYAN}Initializing...{Colors.END}")
        cli.engine = EnhancedRAGEngine()
        cli.processor = RepositoryProcessor(cli.engine)
        print(f"{Colors.GREEN}✓ Ready{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Failed: {e}{Colors.END}")
        return
    
    # Handle single command mode
    if args.stats:
        cli.cmd_stats()
    elif args.process:
        cli.cmd_process(args.process)
    elif args.query:
        cli.cmd_query(args.query)
    else:
        # Interactive mode
        cli.start()


if __name__ == "__main__":
    main()
