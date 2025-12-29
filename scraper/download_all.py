"""
Comprehensive document downloader for all portals and Shasanadesh GOs
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional, Set
from urllib.parse import urlparse, unquote
from dataclasses import dataclass, asdict
import re

import aiohttp
import aiofiles
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from tenacity import retry, stop_after_attempt, wait_exponential
from playwright.async_api import async_playwright

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR, LOGS_DIR, MAX_CONCURRENT_DOWNLOADS

console = Console()


@dataclass
class DownloadResult:
    url: str
    source: str  # 'portal' or 'shasanadesh'
    downloaded: bool = False
    local_path: Optional[str] = None
    file_hash: Optional[str] = None
    error: Optional[str] = None


class ComprehensiveDownloader:
    """Downloads documents from all sources"""

    def __init__(self):
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        self.downloaded_hashes: Set[str] = set()
        self.stats = {
            "portal_success": 0,
            "portal_failed": 0,
            "shasanadesh_success": 0,
            "shasanadesh_failed": 0,
            "duplicates": 0
        }

    def get_safe_filename(self, url: str, prefix: str = "") -> str:
        """Generate a safe filename from URL"""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        filename = Path(path).name

        if not filename or filename == "/" or "aspx" in filename:
            # For Shasanadesh, extract ID from URL
            if "id1=" in url:
                id_match = re.search(r'id1=([^&]+)', url)
                if id_match:
                    filename = f"GO_{id_match.group(1)[:20]}.pdf"
                else:
                    filename = f"{hashlib.md5(url.encode()).hexdigest()[:16]}.pdf"
            else:
                filename = f"{hashlib.md5(url.encode()).hexdigest()[:16]}.pdf"

        # Sanitize filename
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
        filename = "".join(c if c in safe_chars else "_" for c in filename)

        if prefix:
            filename = f"{prefix}_{filename}"

        return filename

    def get_file_hash(self, content: bytes) -> str:
        """Generate MD5 hash of file content"""
        return hashlib.md5(content).hexdigest()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def download_direct(
        self,
        session: aiohttp.ClientSession,
        url: str,
        dest_dir: Path
    ) -> Optional[DownloadResult]:
        """Download a file directly via HTTP"""
        result = DownloadResult(url=url, source="portal")

        async with self.semaphore:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status != 200:
                        result.error = f"HTTP {response.status}"
                        return result

                    content = await response.read()

                    if len(content) < 100:
                        result.error = "File too small"
                        return result

                    file_hash = self.get_file_hash(content)
                    if file_hash in self.downloaded_hashes:
                        self.stats["duplicates"] += 1
                        result.file_hash = file_hash
                        result.error = "Duplicate"
                        return result

                    self.downloaded_hashes.add(file_hash)

                    # Determine file type and directory
                    content_type = response.headers.get('content-type', '').lower()
                    if 'pdf' in content_type or url.lower().endswith('.pdf'):
                        ext = 'pdf'
                    elif 'doc' in content_type or url.lower().endswith(('.doc', '.docx')):
                        ext = 'doc'
                    else:
                        ext = 'pdf'  # Default to pdf

                    doc_dir = dest_dir / ext
                    doc_dir.mkdir(parents=True, exist_ok=True)

                    filename = self.get_safe_filename(url)
                    if not filename.endswith(f'.{ext}'):
                        filename = f"{filename}.{ext}"

                    file_path = doc_dir / filename

                    # Handle conflicts
                    counter = 1
                    original_stem = file_path.stem
                    while file_path.exists():
                        file_path = doc_dir / f"{original_stem}_{counter}{file_path.suffix}"
                        counter += 1

                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(content)

                    result.downloaded = True
                    result.local_path = str(file_path)
                    result.file_hash = file_hash
                    return result

            except Exception as e:
                result.error = str(e)
                return result

    async def download_shasanadesh_page(
        self,
        browser,
        url: str,
        dest_dir: Path
    ) -> DownloadResult:
        """Download PDF from Shasanadesh GO page using Playwright"""
        result = DownloadResult(url=url, source="shasanadesh")

        try:
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                ignore_https_errors=True
            )
            page = await context.new_page()

            # Set up download handling
            download_path = dest_dir / "pdf"
            download_path.mkdir(parents=True, exist_ok=True)

            # Navigate to the GO page
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(1)

            # Try to find and click download/view PDF button
            pdf_content = None
            pdf_url = None

            # Method 1: Look for direct PDF link or iframe
            try:
                # Check for iframe containing PDF
                frames = page.frames
                for frame in frames:
                    frame_url = frame.url
                    if '.pdf' in frame_url.lower():
                        pdf_url = frame_url
                        break

                # Check for object/embed with PDF
                if not pdf_url:
                    pdf_element = await page.query_selector('object[data*=".pdf"], embed[src*=".pdf"], iframe[src*=".pdf"]')
                    if pdf_element:
                        pdf_url = await pdf_element.get_attribute('data') or await pdf_element.get_attribute('src')

                # Check for direct PDF link
                if not pdf_url:
                    pdf_link = await page.query_selector('a[href*=".pdf"]')
                    if pdf_link:
                        pdf_url = await pdf_link.get_attribute('href')

            except Exception:
                pass

            # Method 2: Download button click
            if not pdf_url:
                try:
                    download_btn = await page.query_selector('input[type="submit"], button[onclick*="download"], a[onclick*="download"]')
                    if download_btn:
                        async with page.expect_download(timeout=30000) as download_info:
                            await download_btn.click()
                        download = await download_info.value

                        # Get the downloaded file
                        filename = self.get_safe_filename(url, "shasanadesh")
                        file_path = download_path / filename
                        await download.save_as(file_path)

                        # Read and hash content
                        async with aiofiles.open(file_path, "rb") as f:
                            content = await f.read()

                        file_hash = self.get_file_hash(content)
                        if file_hash in self.downloaded_hashes:
                            os.remove(file_path)
                            self.stats["duplicates"] += 1
                            result.file_hash = file_hash
                            result.error = "Duplicate"
                        else:
                            self.downloaded_hashes.add(file_hash)
                            result.downloaded = True
                            result.local_path = str(file_path)
                            result.file_hash = file_hash

                        await context.close()
                        return result
                except Exception:
                    pass

            # Method 3: Download from PDF URL
            if pdf_url:
                if not pdf_url.startswith('http'):
                    base_url = "https://shasanadesh.up.gov.in"
                    pdf_url = base_url + pdf_url if pdf_url.startswith('/') else base_url + '/' + pdf_url

                async with aiohttp.ClientSession() as session:
                    async with session.get(pdf_url, ssl=False, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status == 200:
                            content = await resp.read()

                            file_hash = self.get_file_hash(content)
                            if file_hash in self.downloaded_hashes:
                                self.stats["duplicates"] += 1
                                result.file_hash = file_hash
                                result.error = "Duplicate"
                            else:
                                self.downloaded_hashes.add(file_hash)

                                filename = self.get_safe_filename(url, "shasanadesh")
                                file_path = download_path / filename

                                counter = 1
                                original_stem = file_path.stem
                                while file_path.exists():
                                    file_path = download_path / f"{original_stem}_{counter}{file_path.suffix}"
                                    counter += 1

                                async with aiofiles.open(file_path, "wb") as f:
                                    await f.write(content)

                                result.downloaded = True
                                result.local_path = str(file_path)
                                result.file_hash = file_hash
            else:
                result.error = "No PDF found on page"

            await context.close()
            return result

        except Exception as e:
            result.error = str(e)
            return result

    async def download_portal_documents(self, docs_file: Path):
        """Download all portal documents"""
        console.print("\n[bold blue]Downloading Portal Documents[/bold blue]")

        if not docs_file.exists():
            console.print(f"[red]File not found: {docs_file}[/red]")
            return

        async with aiofiles.open(docs_file, "r") as f:
            content = await f.read()
            docs = json.loads(content)

        # Filter out already downloaded and non-downloadable URLs
        urls_to_download = []
        for url, data in docs.items():
            if not data.get("downloaded", False):
                # Skip udyogbandhu.com (dead domain)
                if "udyogbandhu.com" not in url:
                    urls_to_download.append(url)

        console.print(f"[cyan]Found {len(urls_to_download)} documents to download[/cyan]")

        if not urls_to_download:
            console.print("[yellow]No documents to download[/yellow]")
            return

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS, ssl=False)
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Downloading portal docs...", total=len(urls_to_download))

                for url in urls_to_download:
                    result = await self.download_direct(session, url, RAW_DIR)
                    if result and result.downloaded:
                        self.stats["portal_success"] += 1
                        # Update the docs dict
                        if url in docs:
                            docs[url]["downloaded"] = True
                            docs[url]["local_path"] = result.local_path
                            docs[url]["file_hash"] = result.file_hash
                    else:
                        self.stats["portal_failed"] += 1

                    progress.update(task, advance=1)

        # Save updated docs
        async with aiofiles.open(docs_file, "w") as f:
            await f.write(json.dumps(docs, indent=2, ensure_ascii=False))

        console.print(f"\n[green]Portal Downloads: {self.stats['portal_success']} success, {self.stats['portal_failed']} failed[/green]")

    async def download_shasanadesh_gos(self, links_file: Path):
        """Download all Shasanadesh GOs"""
        console.print("\n[bold blue]Downloading Shasanadesh GOs[/bold blue]")

        if not links_file.exists():
            console.print(f"[red]File not found: {links_file}[/red]")
            return

        async with aiofiles.open(links_file, "r") as f:
            content = await f.read()

        links = [line.strip() for line in content.strip().split('\n') if line.strip()]
        links = list(set(links))  # Remove duplicates

        console.print(f"[cyan]Found {len(links)} unique Shasanadesh GO links[/cyan]")

        if not links:
            console.print("[yellow]No links to download[/yellow]")
            return

        shasanadesh_dir = RAW_DIR / "shasanadesh"
        shasanadesh_dir.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Downloading Shasanadesh GOs...", total=len(links))

                for url in links:
                    result = await self.download_shasanadesh_page(browser, url, shasanadesh_dir)
                    if result and result.downloaded:
                        self.stats["shasanadesh_success"] += 1
                    else:
                        self.stats["shasanadesh_failed"] += 1
                        if result and result.error:
                            console.print(f"[dim red]{url}: {result.error}[/dim red]")

                    progress.update(task, advance=1)

            await browser.close()

        console.print(f"\n[green]Shasanadesh Downloads: {self.stats['shasanadesh_success']} success, {self.stats['shasanadesh_failed']} failed[/green]")

    async def run_all(self):
        """Run all downloads"""
        console.print("[bold cyan]Starting Comprehensive Document Download[/bold cyan]")

        # Download portal documents
        portal_docs = LOGS_DIR / "all_portal_documents.json"
        await self.download_portal_documents(portal_docs)

        # Download Shasanadesh GOs
        shasanadesh_links = Path("/tmp/shasanadesh_all_links.txt")
        await self.download_shasanadesh_gos(shasanadesh_links)

        # Final summary
        console.print("\n" + "="*50)
        console.print("[bold green]Download Complete![/bold green]")
        console.print(f"  Portal documents: {self.stats['portal_success']} downloaded")
        console.print(f"  Shasanadesh GOs: {self.stats['shasanadesh_success']} downloaded")
        console.print(f"  Duplicates skipped: {self.stats['duplicates']}")
        console.print(f"  Failed: {self.stats['portal_failed'] + self.stats['shasanadesh_failed']}")


async def main():
    downloader = ComprehensiveDownloader()
    await downloader.run_all()


if __name__ == "__main__":
    asyncio.run(main())
