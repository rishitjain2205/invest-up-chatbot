"""
Embedding generation - supports Azure OpenAI and OpenAI
Optimized with TRUE API batching for 20-50x speedup
"""
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

import tiktoken
from openai import AzureOpenAI, OpenAI
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    CHUNKS_DIR
)

console = Console()

# Check for OpenAI API key (for embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "")
AZURE_EMBEDDING_API_VERSION = "2023-05-15"  # Embedding API version

# Max tokens for embedding model
MAX_TOKENS = 8000
# Batch size for API calls (Azure supports up to 2048, but we use smaller for reliability)
API_BATCH_SIZE = 100


class EmbeddingGenerator:
    """Generate embeddings using OpenAI or Azure OpenAI with TRUE batching"""

    def __init__(self):
        self.progress_batch_size = 500  # How often to save progress
        self.api_batch_size = API_BATCH_SIZE  # Texts per API call
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Determine which client to use
        if AZURE_EMBEDDING_DEPLOYMENT:
            console.print("[blue]Using Azure OpenAI for embeddings (BATCHED)[/blue]")
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_EMBEDDING_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            self.model = AZURE_EMBEDDING_DEPLOYMENT
            self.use_azure = True
        elif OPENAI_API_KEY:
            console.print("[blue]Using OpenAI for embeddings (BATCHED)[/blue]")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = "text-embedding-3-large"
            self.use_azure = False
        else:
            console.print("[yellow]Warning: No embedding API configured. Trying Azure...[/yellow]")
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            self.model = "text-embedding-3-large"
            self.use_azure = True

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def truncate_text(self, text: str, max_tokens: int = MAX_TOKENS) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])

    def _embed_batch(self, texts: List[str], retry_count: int = 3) -> List[List[float]]:
        """Embed a batch of texts in a SINGLE API call"""
        # Truncate any texts that exceed token limit
        truncated_texts = [self.truncate_text(t) for t in texts]

        for attempt in range(retry_count):
            try:
                response = self.client.embeddings.create(
                    input=truncated_texts,
                    model=self.model,
                    dimensions=EMBEDDING_DIMENSIONS
                )
                # Return embeddings in the same order as input
                return [item.embedding for item in response.data]
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    wait_time = (attempt + 1) * 5
                    console.print(f"[yellow]Rate limited, waiting {wait_time}s...[/yellow]")
                    time.sleep(wait_time)
                elif attempt < retry_count - 1:
                    console.print(f"[yellow]Error (attempt {attempt+1}): {e}, retrying...[/yellow]")
                    time.sleep(2)
                else:
                    console.print(f"[red]Failed after {retry_count} attempts: {e}[/red]")
                    # Return empty embeddings for this batch
                    return [[] for _ in texts]
        return [[] for _ in texts]

    def generate_all_embeddings(self) -> Dict[str, List[float]]:
        """Generate embeddings for all chunks using TRUE batching"""
        console.print("[bold blue]Generating embeddings with TRUE API batching...[/bold blue]")

        chunks_file = CHUNKS_DIR / "all_chunks.json"
        if not chunks_file.exists():
            console.print("[red]No chunks found. Run chunker first.[/red]")
            return {}

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Check for existing embeddings to resume
        embeddings_file = CHUNKS_DIR / "embeddings.json"
        embeddings = {}
        if embeddings_file.exists():
            try:
                with open(embeddings_file, "r") as f:
                    embeddings = json.load(f)
                console.print(f"[green]Loaded {len(embeddings)} existing embeddings[/green]")
            except:
                console.print("[yellow]Could not load existing embeddings, starting fresh[/yellow]")

        # Filter out already processed chunks
        remaining_chunks = [c for c in chunks if c["chunk_id"] not in embeddings]
        console.print(f"[blue]Processing {len(remaining_chunks)} remaining chunks[/blue]")

        if not remaining_chunks:
            console.print("[green]All chunks already have embeddings![/green]")
            return embeddings

        # Calculate batches
        total_api_calls = (len(remaining_chunks) + self.api_batch_size - 1) // self.api_batch_size
        console.print(f"[cyan]Will make ~{total_api_calls} API calls (batch size: {self.api_batch_size})[/cyan]")

        processed = 0
        start_time = time.time()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Embedding batches...", total=len(remaining_chunks))

            # Process in API batches
            for i in range(0, len(remaining_chunks), self.api_batch_size):
                batch = remaining_chunks[i:i + self.api_batch_size]
                texts = [chunk["text"] for chunk in batch]
                chunk_ids = [chunk["chunk_id"] for chunk in batch]

                # Single API call for entire batch
                batch_embeddings = self._embed_batch(texts)

                # Store results
                for chunk_id, embedding in zip(chunk_ids, batch_embeddings):
                    if embedding:
                        embeddings[chunk_id] = embedding

                processed += len(batch)
                progress.update(task, advance=len(batch))

                # Save progress periodically
                if processed % self.progress_batch_size == 0:
                    self.save_embeddings(embeddings)
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = len(remaining_chunks) - processed
                    eta_seconds = remaining / rate if rate > 0 else 0
                    console.print(f"[dim]Saved {len(embeddings)} embeddings. Rate: {rate:.1f}/s, ETA: {eta_seconds/60:.1f}min[/dim]")

        # Final save
        self.save_embeddings(embeddings)

        elapsed = time.time() - start_time
        console.print(f"\n[bold green]Completed![/bold green]")
        console.print(f"  Total embeddings: {len(embeddings)}")
        console.print(f"  Time taken: {elapsed/60:.1f} minutes")
        console.print(f"  Rate: {len(remaining_chunks)/elapsed:.1f} embeddings/second")

        return embeddings

    def save_embeddings(self, embeddings: Dict[str, List[float]]):
        """Save embeddings to file"""
        output_file = CHUNKS_DIR / "embeddings.json"

        with open(output_file, "w") as f:
            json.dump(embeddings, f)


def main():
    generator = EmbeddingGenerator()
    generator.generate_all_embeddings()


if __name__ == "__main__":
    main()
