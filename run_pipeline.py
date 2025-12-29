#!/usr/bin/env python3
"""
Main orchestration script for Invest UP Chatbot pipeline

Multi-Portal Coverage:
  - invest.up.gov.in
  - niveshmitra.up.nic.in
  - niveshsarathi.up.gov.in
  - startinup.up.gov.in
  - upeida.up.gov.in
  - UPSIDA

Usage:
    python run_pipeline.py crawl      # Crawl all portals
    python run_pipeline.py download   # Download all documents
    python run_pipeline.py extract    # Extract HTML content (included in crawl)
    python run_pipeline.py parse      # Parse PDFs with LlamaParser
    python run_pipeline.py chunk      # Chunk all content
    python run_pipeline.py embed      # Generate embeddings
    python run_pipeline.py index      # Upload to Pinecone
    python run_pipeline.py all        # Run complete pipeline
    python run_pipeline.py serve      # Start API server
    python run_pipeline.py test       # Test the chatbot
"""
import asyncio
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


async def run_crawl():
    """Crawl all portals"""
    console.print(Panel("[bold]Step 1: Multi-Portal Crawling[/bold]\n\nCrawling 6 portals for 100% coverage", style="blue"))
    from scraper.multi_portal_crawler import main
    await main()


async def run_download():
    """Download all documents"""
    console.print(Panel("[bold]Step 2: Downloading Documents[/bold]\n\nDownloading ~1100 documents from all portals", style="blue"))
    from scraper.multi_portal_downloader import main
    await main()


async def run_extract():
    """Extract additional HTML content"""
    console.print(Panel("[bold]Step 3: HTML Content Extraction[/bold]\n\nNote: HTML content is extracted during crawl", style="blue"))
    # Content is already extracted in the multi-portal crawler
    console.print("[green]HTML content was extracted during crawl phase[/green]")


async def run_parse():
    """Parse documents with LlamaParser"""
    console.print(Panel("[bold]Step 4: Parsing Documents with LlamaParser[/bold]\n\nProcessing PDFs, extracting text and tables", style="blue"))
    from processor.llamaparser_client import main
    await main()


def run_chunk():
    """Chunk all content"""
    console.print(Panel("[bold]Step 5: Chunking Content[/bold]\n\nCreating semantic chunks for embedding", style="blue"))
    from processor.chunker import main
    main()


def run_embed():
    """Generate embeddings"""
    console.print(Panel("[bold]Step 6: Generating Embeddings[/bold]\n\nUsing Azure OpenAI text-embedding-3-large", style="blue"))
    from rag.embeddings import main
    main()


def run_index():
    """Upload to Pinecone"""
    console.print(Panel("[bold]Step 7: Indexing to Pinecone[/bold]\n\nUploading vectors for retrieval", style="blue"))
    from rag.vector_store import main
    main()


def run_serve():
    """Start API server"""
    console.print(Panel("[bold]Starting API Server[/bold]\n\nAPI running on http://localhost:8000", style="green"))
    from api.main import run_server
    run_server()


def run_test():
    """Test the chatbot"""
    console.print(Panel("[bold]Testing Chatbot[/bold]", style="green"))
    from rag.retriever import main
    main()


async def run_shasanadesh():
    """Run Shasanadesh crawler (requires manual CAPTCHA)"""
    console.print(Panel(
        "[bold]Shasanadesh Historical GO Crawler[/bold]\n\n"
        "This will crawl ~1500 historical GOs from Industrial Development Dept\n"
        "Date Range: 2015-2025\n\n"
        "[yellow]You will need to solve ONE CAPTCHA manually[/yellow]",
        style="blue"
    ))
    from scraper.shasanadesh_crawler import main
    await main()


async def run_shasanadesh_download():
    """Download Shasanadesh GOs"""
    console.print(Panel("[bold]Downloading Shasanadesh GOs[/bold]", style="blue"))
    from scraper.shasanadesh_crawler import ShasanadeshDownloader
    downloader = ShasanadeshDownloader()
    await downloader.download_all()


async def run_all():
    """Run complete pipeline"""
    console.print(Panel(
        "[bold]Running Complete Pipeline[/bold]\n\n"
        "Portals: invest.up.gov.in, niveshmitra.up.nic.in, niveshsarathi.up.gov.in,\n"
        "         startinup.up.gov.in, upeida.up.gov.in, UPSIDA\n\n"
        "Estimated: ~1,100 documents, ~250 HTML pages",
        style="green"
    ))

    try:
        # Step 1: Crawl all portals
        await run_crawl()

        # Step 2: Download all documents
        await run_download()

        # Step 3: HTML extraction (done in crawl)
        await run_extract()

        # Step 4: Parse with LlamaParser
        await run_parse()

        # Step 5: Chunk content
        run_chunk()

        # Step 6: Generate embeddings
        run_embed()

        # Step 7: Index to Pinecone
        run_index()

        console.print("\n")
        console.print(Panel(
            "[bold green]Pipeline Complete![/bold green]\n\n"
            "Your Invest UP chatbot is ready!\n\n"
            "To start the chatbot:\n"
            "  python run_pipeline.py serve\n\n"
            "Then open frontend/index.html in your browser\n\n"
            "Or test via CLI:\n"
            "  python run_pipeline.py test",
            title="Success"
        ))

    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise


def show_status():
    """Show current pipeline status"""
    from config import LOGS_DIR, RAW_DIR, PROCESSED_DIR, CHUNKS_DIR

    console.print(Panel("[bold]Pipeline Status[/bold]", style="blue"))

    # Check crawl results
    crawl_file = LOGS_DIR / "all_portal_documents.json"
    if crawl_file.exists():
        with open(crawl_file) as f:
            docs = json.load(f)
        console.print(f"[green]Crawl: {len(docs)} documents discovered[/green]")
    else:
        console.print("[yellow]Crawl: Not run yet[/yellow]")

    # Check downloads
    download_file = LOGS_DIR / "downloaded_documents.json"
    if download_file.exists():
        with open(download_file) as f:
            docs = json.load(f)
        downloaded = sum(1 for d in docs.values() if d.get("downloaded"))
        console.print(f"[green]Download: {downloaded} documents downloaded[/green]")
    else:
        console.print("[yellow]Download: Not run yet[/yellow]")

    # Check parsed
    parsed_file = PROCESSED_DIR / "parsed_documents.json"
    if parsed_file.exists():
        with open(parsed_file) as f:
            docs = json.load(f)
        console.print(f"[green]Parse: {len(docs)} documents parsed[/green]")
    else:
        console.print("[yellow]Parse: Not run yet[/yellow]")

    # Check chunks
    chunks_file = CHUNKS_DIR / "all_chunks.json"
    if chunks_file.exists():
        with open(chunks_file) as f:
            chunks = json.load(f)
        console.print(f"[green]Chunk: {len(chunks)} chunks created[/green]")
    else:
        console.print("[yellow]Chunk: Not run yet[/yellow]")

    # Check embeddings
    embeddings_file = CHUNKS_DIR / "embeddings.json"
    if embeddings_file.exists():
        with open(embeddings_file) as f:
            embeddings = json.load(f)
        console.print(f"[green]Embed: {len(embeddings)} embeddings generated[/green]")
    else:
        console.print("[yellow]Embed: Not run yet[/yellow]")


def main():
    if len(sys.argv) < 2:
        console.print(__doc__)
        console.print("\n")
        show_status()
        return

    command = sys.argv[1].lower()

    commands = {
        "crawl": lambda: asyncio.run(run_crawl()),
        "download": lambda: asyncio.run(run_download()),
        "extract": lambda: asyncio.run(run_extract()),
        "parse": lambda: asyncio.run(run_parse()),
        "chunk": run_chunk,
        "embed": run_embed,
        "index": run_index,
        "all": lambda: asyncio.run(run_all()),
        "serve": run_serve,
        "test": run_test,
        "status": show_status,
        "shasanadesh": lambda: asyncio.run(run_shasanadesh()),
        "shasanadesh-download": lambda: asyncio.run(run_shasanadesh_download()),
    }

    if command not in commands:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print(__doc__)
        return

    commands[command]()


if __name__ == "__main__":
    main()
