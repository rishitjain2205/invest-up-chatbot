"""
Document downloader with retry logic and deduplication
"""
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import asdict
from urllib.parse import urlparse, unquote

import aiohttp
import aiofiles
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR, LOGS_DIR, MAX_CONCURRENT_DOWNLOADS
from scraper.crawler import CrawlResult

console = Console()


class DocumentDownloader:
    """Downloads discovered documents with deduplication"""

    def __init__(self):
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        self.downloaded_hashes: set = set()
        self.download_stats = {
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "duplicates": 0
        }

    def get_safe_filename(self, url: str) -> str:
        """Generate a safe filename from URL"""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        filename = Path(path).name

        # Handle URLs without proper filename
        if not filename or filename == "/":
            filename = hashlib.md5(url.encode()).hexdigest()[:16]

        # Sanitize filename
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
        filename = "".join(c if c in safe_chars else "_" for c in filename)

        return filename

    def get_file_hash(self, content: bytes) -> str:
        """Generate MD5 hash of file content"""
        return hashlib.md5(content).hexdigest()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        result: CrawlResult
    ) -> Optional[str]:
        """Download a single file with retry logic"""
        async with self.semaphore:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        console.print(f"[yellow]HTTP {response.status} for {url}[/yellow]")
                        return None

                    content = await response.read()

                    # Check for duplicates
                    file_hash = self.get_file_hash(content)
                    if file_hash in self.downloaded_hashes:
                        console.print(f"[yellow]Duplicate (skipped): {url}[/yellow]")
                        self.download_stats["duplicates"] += 1
                        result.file_hash = file_hash
                        result.metadata["duplicate"] = True
                        return None

                    self.downloaded_hashes.add(file_hash)

                    # Create directory structure based on document type
                    doc_dir = RAW_DIR / result.doc_type
                    doc_dir.mkdir(parents=True, exist_ok=True)

                    # Generate unique filename
                    filename = self.get_safe_filename(url)
                    file_path = doc_dir / filename

                    # Handle filename conflicts
                    counter = 1
                    original_stem = file_path.stem
                    while file_path.exists():
                        file_path = doc_dir / f"{original_stem}_{counter}{file_path.suffix}"
                        counter += 1

                    # Save file
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(content)

                    result.downloaded = True
                    result.local_path = str(file_path)
                    result.file_hash = file_hash
                    result.metadata["file_size"] = len(content)

                    return str(file_path)

            except asyncio.TimeoutError:
                console.print(f"[red]Timeout downloading {url}[/red]")
                raise
            except Exception as e:
                console.print(f"[red]Error downloading {url}: {e}[/red]")
                raise

    async def download_all(self, crawl_results: Dict[str, CrawlResult]) -> Dict[str, CrawlResult]:
        """Download all documents from crawl results"""
        console.print(f"\n[bold blue]Downloading {len(crawl_results)} documents...[/bold blue]")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS, ssl=False)
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Downloading...", total=len(crawl_results))

                tasks = []
                for url, result in crawl_results.items():
                    task_coro = self.download_with_progress(
                        session, url, result, progress, task
                    )
                    tasks.append(task_coro)

                await asyncio.gather(*tasks, return_exceptions=True)

        # Save updated results
        await self.save_download_results(crawl_results)

        console.print(f"\n[bold green]Download Summary:[/bold green]")
        console.print(f"  Successful: {self.download_stats['success']}")
        console.print(f"  Failed: {self.download_stats['failed']}")
        console.print(f"  Duplicates: {self.download_stats['duplicates']}")

        return crawl_results

    async def download_with_progress(
        self,
        session: aiohttp.ClientSession,
        url: str,
        result: CrawlResult,
        progress: Progress,
        task
    ):
        """Download with progress tracking"""
        try:
            path = await self.download_file(session, url, result)
            if path:
                self.download_stats["success"] += 1
        except Exception:
            self.download_stats["failed"] += 1
        finally:
            progress.update(task, advance=1)

    async def save_download_results(self, crawl_results: Dict[str, CrawlResult]):
        """Save updated results with download info"""
        results_file = LOGS_DIR / "download_results.json"
        results_data = {url: asdict(result) for url, result in crawl_results.items()}

        async with aiofiles.open(results_file, "w") as f:
            await f.write(json.dumps(results_data, indent=2, ensure_ascii=False))

        console.print(f"[blue]Download results saved to {results_file}[/blue]")


async def main():
    """Load crawl results and download all documents"""
    crawl_file = LOGS_DIR / "crawl_results.json"

    if not crawl_file.exists():
        console.print("[red]No crawl results found. Run crawler first.[/red]")
        return

    async with aiofiles.open(crawl_file, "r") as f:
        content = await f.read()
        crawl_data = json.loads(content)

    # Convert to CrawlResult objects
    crawl_results = {}
    for url, data in crawl_data.items():
        crawl_results[url] = CrawlResult(**data)

    downloader = DocumentDownloader()
    await downloader.download_all(crawl_results)


if __name__ == "__main__":
    asyncio.run(main())
