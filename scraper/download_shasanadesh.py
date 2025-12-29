"""
Shasanadesh GO Downloader - Handles direct PDF download links
"""
import asyncio
import hashlib
import os
from pathlib import Path
from typing import Set
import aiofiles
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR

console = Console()


class ShasanadeshDownloader:
    """Downloads Shasanadesh GOs using Playwright with download handling"""

    def __init__(self):
        self.downloaded_hashes: Set[str] = set()
        self.stats = {"success": 0, "failed": 0, "duplicates": 0}
        self.download_dir = RAW_DIR / "shasanadesh"
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def get_file_hash(self, filepath: Path) -> str:
        """Generate MD5 hash of file content"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    async def download_go(self, context, url: str, index: int) -> bool:
        """Download a single GO PDF"""
        page = None
        try:
            page = await context.new_page()

            # Extract ID from URL for naming
            id_part = url.split('id1=')[-1].replace('=', '').replace('/', '_')[:30]
            filename = f"GO_{index:04d}_{id_part}.pdf"
            download_path = self.download_dir / filename

            # Handle the download event
            download = None

            async def handle_download(d):
                nonlocal download
                download = d

            page.on("download", handle_download)

            try:
                # Navigate - will either load page or trigger download
                response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as nav_error:
                # Navigation may fail when download starts - this is expected
                error_msg = str(nav_error).lower()
                if "download" not in error_msg:
                    # Real navigation error
                    self.stats["failed"] += 1
                    if page:
                        await page.close()
                    return False

            # Wait a bit for download to be captured
            await asyncio.sleep(2)

            if download:
                try:
                    await download.save_as(download_path)
                    await asyncio.sleep(0.5)

                    if download_path.exists() and download_path.stat().st_size > 100:
                        file_hash = self.get_file_hash(download_path)
                        if file_hash in self.downloaded_hashes:
                            os.remove(download_path)
                            self.stats["duplicates"] += 1
                        else:
                            self.downloaded_hashes.add(file_hash)
                            self.stats["success"] += 1
                            if page:
                                await page.close()
                            return True
                    else:
                        if download_path.exists():
                            os.remove(download_path)
                        self.stats["failed"] += 1
                except Exception as save_error:
                    self.stats["failed"] += 1
            else:
                # No download triggered, try to find PDF on page
                try:
                    # Check for embedded PDF object/iframe
                    pdf_src = await page.evaluate('''() => {
                        const obj = document.querySelector('object[data], embed[src], iframe[src]');
                        if (obj) {
                            return obj.getAttribute('data') || obj.getAttribute('src');
                        }
                        const link = document.querySelector('a[href*=".pdf"]');
                        if (link) {
                            return link.getAttribute('href');
                        }
                        return null;
                    }''')

                    if pdf_src:
                        if not pdf_src.startswith('http'):
                            pdf_src = f"https://shasanadesh.up.gov.in{pdf_src}" if pdf_src.startswith('/') else f"https://shasanadesh.up.gov.in/{pdf_src}"

                        # Try to download the PDF directly
                        import aiohttp
                        async with aiohttp.ClientSession() as session:
                            async with session.get(pdf_src, ssl=False, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                                if resp.status == 200:
                                    content = await resp.read()
                                    if len(content) > 100:
                                        file_hash = hashlib.md5(content).hexdigest()
                                        if file_hash not in self.downloaded_hashes:
                                            self.downloaded_hashes.add(file_hash)
                                            async with aiofiles.open(download_path, 'wb') as f:
                                                await f.write(content)
                                            self.stats["success"] += 1
                                            if page:
                                                await page.close()
                                            return True
                                        else:
                                            self.stats["duplicates"] += 1
                    else:
                        self.stats["failed"] += 1
                except Exception:
                    self.stats["failed"] += 1

            if page:
                await page.close()
            return False

        except Exception as e:
            self.stats["failed"] += 1
            if page:
                try:
                    await page.close()
                except:
                    pass
            return False

    async def download_all(self, links_file: Path):
        """Download all Shasanadesh GOs"""
        console.print("\n[bold cyan]Downloading Shasanadesh GOs[/bold cyan]")

        if not links_file.exists():
            console.print(f"[red]File not found: {links_file}[/red]")
            return

        async with aiofiles.open(links_file, "r") as f:
            content = await f.read()

        links = list(set([line.strip() for line in content.strip().split('\n') if line.strip()]))
        console.print(f"[cyan]Found {len(links)} unique links[/cyan]")

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled']
            )

            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                accept_downloads=True,
                ignore_https_errors=True
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Downloading...", total=len(links))

                for i, url in enumerate(links):
                    await self.download_go(context, url, i)
                    progress.update(task, advance=1)

            await context.close()
            await browser.close()

        console.print(f"\n[bold green]Download Complete![/bold green]")
        console.print(f"  Success: {self.stats['success']}")
        console.print(f"  Failed: {self.stats['failed']}")
        console.print(f"  Duplicates: {self.stats['duplicates']}")


async def main():
    links_file = Path("/tmp/shasanadesh_batch6.txt")
    downloader = ShasanadeshDownloader()
    await downloader.download_all(links_file)


if __name__ == "__main__":
    asyncio.run(main())
