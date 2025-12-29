"""
Pinecone vector store for document retrieval
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
import time

from pinecone import Pinecone, ServerlessSpec
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSIONS,
    CHUNKS_DIR
)

console = Console()


class PineconeVectorStore:
    """Pinecone vector store for document storage and retrieval"""

    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.index = None
        self.batch_size = 100

    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            console.print(f"[blue]Creating index: {self.index_name}[/blue]")
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSIONS,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready with proper polling
            console.print("[yellow]Waiting for index to be ready...[/yellow]")
            max_wait = 300  # 5 minutes max
            waited = 0
            while waited < max_wait:
                try:
                    index_desc = self.pc.describe_index(self.index_name)
                    if index_desc.status.ready:
                        console.print("[green]Index is ready![/green]")
                        break
                except Exception:
                    pass
                time.sleep(5)
                waited += 5
                if waited % 30 == 0:
                    console.print(f"[yellow]Still waiting... ({waited}s)[/yellow]")
        else:
            console.print(f"[green]Index {self.index_name} already exists[/green]")

        self.index = self.pc.Index(self.index_name)

    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        if not self.index:
            self.index = self.pc.Index(self.index_name)
        return self.index.describe_index_stats()

    def upsert_chunks(self) -> int:
        """Upsert all chunks with embeddings to Pinecone"""
        console.print("[bold blue]Upserting chunks to Pinecone...[/bold blue]")

        # Create index if needed
        self.create_index()

        # Load chunks
        chunks_file = CHUNKS_DIR / "all_chunks.json"
        embeddings_file = CHUNKS_DIR / "embeddings.json"

        if not chunks_file.exists() or not embeddings_file.exists():
            console.print("[red]Missing chunks or embeddings. Run previous steps first.[/red]")
            return 0

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        with open(embeddings_file, "r") as f:
            embeddings = json.load(f)

        # Prepare vectors for upsert
        vectors = []
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in embeddings:
                continue

            vector = {
                "id": chunk_id,
                "values": embeddings[chunk_id],
                "metadata": {
                    "text": chunk["text"][:8000],  # Pinecone metadata limit
                    "source_url": chunk["source_url"],
                    "source_file": chunk["source_file"],
                    "portal": chunk.get("portal", "unknown"),
                    "doc_type": chunk["doc_type"],
                    "section": chunk["section"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"]
                }
            }
            vectors.append(vector)

        if not vectors:
            console.print("[yellow]No vectors to upsert[/yellow]")
            return 0

        # Upsert in batches
        total_batches = (len(vectors) + self.batch_size - 1) // self.batch_size
        upserted = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as progress:
            task = progress.add_task("Upserting...", total=total_batches)

            for i in range(0, len(vectors), self.batch_size):
                batch = vectors[i:i + self.batch_size]
                try:
                    self.index.upsert(vectors=batch)
                    upserted += len(batch)
                except Exception as e:
                    console.print(f"[red]Error upserting batch: {e}[/red]")

                progress.update(task, advance=1)

        console.print(f"[green]Upserted {upserted} vectors to Pinecone[/green]")

        # Show stats
        stats = self.get_index_stats()
        console.print(f"[blue]Index stats: {stats}[/blue]")

        return upserted

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar documents"""
        if not self.index:
            self.index = self.pc.Index(self.index_name)

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )

        return [
            {
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "source_url": match.metadata.get("source_url", ""),
                "source_file": match.metadata.get("source_file", ""),
                "portal": match.metadata.get("portal", ""),
                "section": match.metadata.get("section", ""),
                "doc_type": match.metadata.get("doc_type", "")
            }
            for match in results.matches
        ]

    def delete_all(self):
        """Delete all vectors from index"""
        if not self.index:
            self.index = self.pc.Index(self.index_name)

        self.index.delete(delete_all=True)
        console.print("[yellow]Deleted all vectors from index[/yellow]")


def main():
    store = PineconeVectorStore()
    store.upsert_chunks()


if __name__ == "__main__":
    main()
