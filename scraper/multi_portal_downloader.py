"""
Document downloader for multi-portal crawl results
"""
import asyncio
import hashlib
import json
import ssl
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse, unquote

import aiohttp
import aiofiles
import certifi
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR, LOGS_DIR, MAX_CONCURRENT_DOWNLOADS

console = Console()


class MultiPortalDownloader:
    """Downloads documents from all crawled portals"""

    def __init__(self):
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        self.downloaded_hashes: set = set()
        self.stats = {
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "duplicates": 0
        }
        self.failed_urls: list = []

    def get_safe_filename(self, url: str, portal: str) -> str:
        """Generate a safe filename from URL"""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        filename = Path(path).name

        if not filename or filename == "/":
            filename = hashlib.md5(url.encode()).hexdigest()[:16]

        # Sanitize
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
        filename = "".join(c if c in safe_chars else "_" for c in filename)

        # Prefix with portal name for organization
        portal_prefix = portal.replace(".", "_").replace("/", "_")[:20]
        return f"{portal_prefix}__{filename}"

    def get_file_hash(self, content: bytes) -> str:
        """Generate MD5 hash"""
        return hashlib.md5(content).hexdigest()

    async def download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        doc_info: dict
    ) -> Optional[str]:
        """Download a single file"""
        async with self.semaphore:
            try:
                # Create SSL context that ignores certificate errors for gov sites
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=120),
                    ssl=ssl_context
                ) as response:
                    if response.status != 200:
                        console.print(f"[yellow]HTTP {response.status}: {url}[/yellow]")
                        self.failed_urls.append({"url": url, "error": f"HTTP {response.status}"})
                        return None

                    content = await response.read()

                    if len(content) < 100:
                        console.print(f"[yellow]Empty/small file: {url}[/yellow]")
                        return None

                    # Check for duplicates
                    file_hash = self.get_file_hash(content)
                    if file_hash in self.downloaded_hashes:
                        self.stats["duplicates"] += 1
                        doc_info["file_hash"] = file_hash
                        doc_info["duplicate"] = True
                        return None

                    self.downloaded_hashes.add(file_hash)

                    # Organize by portal and doc_type
                    portal = doc_info.get("portal", "unknown")
                    doc_type = doc_info.get("doc_type", "pdf")

                    doc_dir = RAW_DIR / portal.replace(".", "_") / doc_type
                    doc_dir.mkdir(parents=True, exist_ok=True)

                    filename = self.get_safe_filename(url, portal)
                    file_path = doc_dir / filename

                    # Handle conflicts
                    counter = 1
                    original_stem = file_path.stem
                    while file_path.exists():
                        file_path = doc_dir / f"{original_stem}_{counter}{file_path.suffix}"
                        counter += 1

                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(content)

                    doc_info["downloaded"] = True
                    doc_info["local_path"] = str(file_path)
                    doc_info["file_hash"] = file_hash
                    doc_info["file_size"] = len(content)

                    return str(file_path)

            except asyncio.TimeoutError:
                console.print(f"[red]Timeout: {url}[/red]")
                self.failed_urls.append({"url": url, "error": "Timeout"})
                return None
            except Exception as e:
                console.print(f"[red]Error downloading {url}: {e}[/red]")
                self.failed_urls.append({"url": url, "error": str(e)})
                return None

    async def download_all(self) -> Dict:
        """Download all documents from crawl results"""
        # Load crawl results
        docs_file = LOGS_DIR / "all_portal_documents.json"

        if not docs_file.exists():
            console.print("[red]No crawl results found. Run multi-portal crawler first.[/red]")
            return {}

        async with aiofiles.open(docs_file, 'r') as f:
            content = await f.read()
            all_docs = json.loads(content)

        console.print(f"\n[bold blue]Downloading {len(all_docs)} documents...[/bold blue]")

        # Group by portal for stats
        by_portal = {}
        for url, doc in all_docs.items():
            portal = doc.get("portal", "unknown")
            by_portal[portal] = by_portal.get(portal, 0) + 1

        console.print("\n[bold]Documents to download by portal:[/bold]")
        for portal, count in sorted(by_portal.items()):
            console.print(f"  {portal}: {count}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
        }

        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS, ssl=False)
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("{task.completed}/{task.total}"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Downloading...", total=len(all_docs))

                # Process in batches
                batch_size = 20
                urls = list(all_docs.keys())

                for i in range(0, len(urls), batch_size):
                    batch_urls = urls[i:i + batch_size]
                    tasks = []

                    for url in batch_urls:
                        doc_info = all_docs[url]
                        tasks.append(self.download_with_progress(
                            session, url, doc_info, progress, task
                        ))

                    await asyncio.gather(*tasks, return_exceptions=True)

        # Save updated results
        await self.save_results(all_docs)

        # Print summary
        console.print(f"\n[bold green]Download Summary:[/bold green]")
        console.print(f"  Successful: {self.stats['success']}")
        console.print(f"  Failed: {self.stats['failed']}")
        console.print(f"  Duplicates: {self.stats['duplicates']}")

        if self.failed_urls:
            console.print(f"\n[yellow]Failed URLs saved to {LOGS_DIR}/failed_downloads.json[/yellow]")

        return all_docs

    async def download_with_progress(self, session, url, doc_info, progress, task):
        """Download with progress update"""
        try:
            path = await self.download_file(session, url, doc_info)
            if path:
                self.stats["success"] += 1
            else:
                self.stats["failed"] += 1
        except Exception as e:
            self.stats["failed"] += 1
        finally:
            progress.update(task, advance=1)

    async def save_results(self, all_docs: Dict):
        """Save updated results"""
        # Save updated documents
        docs_file = LOGS_DIR / "downloaded_documents.json"
        async with aiofiles.open(docs_file, 'w') as f:
            await f.write(json.dumps(all_docs, indent=2, ensure_ascii=False))

        # Save failed URLs
        if self.failed_urls:
            failed_file = LOGS_DIR / "failed_downloads.json"
            async with aiofiles.open(failed_file, 'w') as f:
                await f.write(json.dumps(self.failed_urls, indent=2))

        console.print(f"[blue]Results saved to {docs_file}[/blue]")


async def main():
    downloader = MultiPortalDownloader()
    await downloader.download_all()


if __name__ == "__main__":
    asyncio.run(main())
