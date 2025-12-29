"""
Comprehensive web crawler for invest.up.gov.in
Uses Playwright for JS rendering and network interception
"""
import asyncio
import json
import re
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
from dataclasses import dataclass, field, asdict
from typing import Set, Dict, List, Optional
from datetime import datetime

from playwright.async_api import async_playwright, Page, Browser, Route
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, TaskID
import aiohttp
import aiofiles

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    BASE_URL, RAW_DIR, LOGS_DIR,
    DOCUMENT_EXTENSIONS, MAX_CONCURRENT_DOWNLOADS, REQUEST_DELAY
)

console = Console()


@dataclass
class CrawlResult:
    """Stores metadata about a discovered document or page"""
    url: str
    source_page: str
    doc_type: str  # 'pdf', 'html', 'xlsx', etc.
    title: Optional[str] = None
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    downloaded: bool = False
    local_path: Optional[str] = None
    file_hash: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class InvestUPCrawler:
    """
    Multi-strategy crawler for 100% document coverage

    Strategies:
    1. Recursive link crawling (BFS)
    2. Network request interception
    3. JavaScript variable extraction
    4. Pagination exhaustion
    5. WordPress upload pattern scanning
    """

    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.document_urls: Set[str] = set()
        self.crawl_results: Dict[str, CrawlResult] = {}
        self.url_queue: asyncio.Queue = asyncio.Queue()
        self.download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

        # Patterns to extract document URLs from JavaScript
        self.js_url_patterns = [
            r'["\']([^"\']*\.pdf)["\']',
            r'["\']([^"\']*\.xlsx?)["\']',
            r'["\']([^"\']*\.docx?)["\']',
            r'["\']([^"\']*\.pptx?)["\']',
            r'href\s*[=:]\s*["\']([^"\']+)["\']',
            r'url\s*[=:]\s*["\']([^"\']+)["\']',
            r'src\s*[=:]\s*["\']([^"\']+)["\']',
        ]

        # Known section URLs for systematic crawling
        self.priority_sections = [
            "/",
            "/policies/",
            "/gos/",
            "/circulars/",
            "/circulars-archive/",
            "/faqs/",
            "/sectors-in-uttar-pradesh/",
            "/economic-snapshot/",
            "/demography/",
            "/infrastructure/",
            "/business-climate/",
            "/newsletter/",
            "/press-releases/",
            "/tender/",
            "/rti/",
            "/model-templates/",
            "/notifications/",
        ]

        # Known sector pages
        self.sector_pages = [
            "/retail-e-commerce/",
            "/defence-aerospace-sector/",
            "/it-ites/",
            "/startup-sector/",
            "/tourism-hospitality/",
            "/msme-sector/",
            "/electronics-technology/",
            "/handloom-and-textile-sector/",
            "/renewable-energy-sector/",
            "/warehousing-logistics-sector/",
            "/auto-components-sector/",
            "/medical-device-and-pharma/",
            "/agriculture-food-processing/",
            "/fintech-sector/",
            "/dairy-sector/",
            "/ev-manufacturing-sector/",
            "/semiconductor-sector/",
            "/film-sector/",
            "/urban-development/",
            "/civil-aviation-sector/",
        ]

    def normalize_url(self, url: str, base_url: str = BASE_URL) -> Optional[str]:
        """Normalize URL and filter out non-relevant URLs"""
        if not url:
            return None

        # Handle relative URLs
        if url.startswith("/"):
            url = urljoin(base_url, url)
        elif not url.startswith("http"):
            url = urljoin(base_url, url)

        # Normalize http to https
        url = url.replace("http://invest.up.gov.in", "https://invest.up.gov.in")

        # Parse and validate
        parsed = urlparse(url)

        # Only crawl invest.up.gov.in
        if "invest.up.gov.in" not in parsed.netloc:
            return None

        # Remove fragments
        url = url.split("#")[0]

        # Remove trailing slash for consistency (except root)
        if url.endswith("/") and url != f"{BASE_URL}/":
            url = url.rstrip("/")

        return url

    def is_document_url(self, url: str) -> bool:
        """Check if URL points to a downloadable document"""
        parsed = urlparse(url.lower())
        path = parsed.path
        return any(path.endswith(ext) for ext in DOCUMENT_EXTENSIONS)

    def get_document_type(self, url: str) -> str:
        """Extract document type from URL"""
        parsed = urlparse(url.lower())
        for ext in DOCUMENT_EXTENSIONS:
            if parsed.path.endswith(ext):
                return ext.lstrip(".")
        return "unknown"

    async def extract_links_from_page(self, page: Page, current_url: str) -> Set[str]:
        """Extract all links from a page using multiple strategies"""
        links = set()

        try:
            content = await page.content()
            soup = BeautifulSoup(content, "lxml")

            # Strategy 1: Standard href links
            for a_tag in soup.find_all("a", href=True):
                href = a_tag.get("href", "")
                normalized = self.normalize_url(href)
                if normalized:
                    links.add(normalized)

            # Strategy 2: iframe sources
            for iframe in soup.find_all("iframe", src=True):
                src = iframe.get("src", "")
                normalized = self.normalize_url(src)
                if normalized:
                    links.add(normalized)

            # Strategy 3: object/embed tags (for embedded PDFs)
            for obj in soup.find_all(["object", "embed"], {"data": True}):
                data = obj.get("data", "") or obj.get("src", "")
                normalized = self.normalize_url(data)
                if normalized:
                    links.add(normalized)

            # Strategy 4: data-* attributes
            for tag in soup.find_all(attrs={"data-href": True}):
                normalized = self.normalize_url(tag["data-href"])
                if normalized:
                    links.add(normalized)
            for tag in soup.find_all(attrs={"data-url": True}):
                normalized = self.normalize_url(tag["data-url"])
                if normalized:
                    links.add(normalized)
            for tag in soup.find_all(attrs={"data-pdf": True}):
                normalized = self.normalize_url(tag["data-pdf"])
                if normalized:
                    links.add(normalized)

            # Strategy 5: onclick handlers
            for tag in soup.find_all(onclick=True):
                onclick = tag.get("onclick", "")
                # Extract URLs from onclick
                for pattern in self.js_url_patterns:
                    matches = re.findall(pattern, onclick)
                    for match in matches:
                        normalized = self.normalize_url(match)
                        if normalized:
                            links.add(normalized)

            # Strategy 6: JavaScript variables in script tags
            for script in soup.find_all("script"):
                if script.string:
                    for pattern in self.js_url_patterns:
                        matches = re.findall(pattern, script.string)
                        for match in matches:
                            if "/wp-content/" in match or match.endswith(tuple(DOCUMENT_EXTENSIONS)):
                                normalized = self.normalize_url(match)
                                if normalized:
                                    links.add(normalized)

            # Strategy 7: Meta refresh redirects
            for meta in soup.find_all("meta", {"http-equiv": "refresh"}):
                content = meta.get("content", "")
                if "url=" in content.lower():
                    url_part = content.lower().split("url=")[-1]
                    normalized = self.normalize_url(url_part.strip("'\""))
                    if normalized:
                        links.add(normalized)

        except Exception as e:
            console.print(f"[yellow]Warning: Error extracting links from {current_url}: {e}[/yellow]")

        return links

    async def detect_pagination(self, page: Page, current_url: str) -> List[str]:
        """Detect and extract all pagination URLs"""
        pagination_urls = []

        try:
            content = await page.content()
            soup = BeautifulSoup(content, "lxml")

            # Pattern 1: ?pagenum=N
            pagenum_links = soup.find_all("a", href=re.compile(r"pagenum=\d+"))
            for link in pagenum_links:
                href = link.get("href", "")
                normalized = self.normalize_url(href)
                if normalized:
                    pagination_urls.append(normalized)

            # Find max page number
            max_page = 1
            for link in pagenum_links:
                href = link.get("href", "")
                match = re.search(r"pagenum=(\d+)", href)
                if match:
                    max_page = max(max_page, int(match.group(1)))

            # Generate all pagination URLs
            if max_page > 1:
                base_path = current_url.split("?")[0]
                for i in range(1, max_page + 1):
                    pagination_urls.append(f"{base_path}?pagenum={i}")

            # Pattern 2: /page/N/
            page_links = soup.find_all("a", href=re.compile(r"/page/\d+"))
            for link in page_links:
                href = link.get("href", "")
                normalized = self.normalize_url(href)
                if normalized:
                    pagination_urls.append(normalized)

        except Exception as e:
            console.print(f"[yellow]Warning: Error detecting pagination: {e}[/yellow]")

        return list(set(pagination_urls))

    async def crawl_page(self, browser: Browser, url: str, progress: Progress, task_id: TaskID) -> Set[str]:
        """Crawl a single page and extract all information"""
        discovered_links = set()

        if url in self.visited_urls:
            return discovered_links

        self.visited_urls.add(url)
        progress.update(task_id, advance=1, description=f"Crawling: {url[:60]}...")

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()

        # Intercept network requests to catch dynamically loaded documents
        intercepted_docs = []

        async def handle_route(route: Route):
            request = route.request
            url_lower = request.url.lower()
            if any(url_lower.endswith(ext) for ext in DOCUMENT_EXTENSIONS):
                intercepted_docs.append(request.url)
            await route.continue_()

        await page.route("**/*", handle_route)

        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(REQUEST_DELAY)  # Polite delay

            # Extract all links
            links = await self.extract_links_from_page(page, url)
            discovered_links.update(links)

            # Check for pagination and add all pages
            pagination_urls = await self.detect_pagination(page, url)
            discovered_links.update(pagination_urls)

            # Add intercepted documents
            for doc_url in intercepted_docs:
                normalized = self.normalize_url(doc_url)
                if normalized:
                    discovered_links.add(normalized)

            # Extract page title for metadata
            title = await page.title()

            # Process discovered links
            for link in discovered_links:
                if self.is_document_url(link):
                    if link not in self.document_urls:
                        self.document_urls.add(link)
                        self.crawl_results[link] = CrawlResult(
                            url=link,
                            source_page=url,
                            doc_type=self.get_document_type(link),
                            title=title,
                            metadata={"source_page_title": title}
                        )
                        console.print(f"[green]Found document: {link}[/green]")

        except Exception as e:
            console.print(f"[red]Error crawling {url}: {e}[/red]")
        finally:
            await context.close()

        return discovered_links

    async def run_crawler(self) -> Dict[str, CrawlResult]:
        """Main crawler execution"""
        console.print("[bold blue]Starting comprehensive crawl of invest.up.gov.in[/bold blue]")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            # Initialize queue with priority sections
            all_sections = self.priority_sections + self.sector_pages
            for section in all_sections:
                url = urljoin(BASE_URL, section)
                await self.url_queue.put(url)

            with Progress() as progress:
                task_id = progress.add_task("[cyan]Crawling...", total=None)

                while not self.url_queue.empty():
                    url = await self.url_queue.get()

                    if url in self.visited_urls:
                        continue

                    discovered = await self.crawl_page(browser, url, progress, task_id)

                    # Add new URLs to queue
                    for link in discovered:
                        if link not in self.visited_urls and not self.is_document_url(link):
                            # Only queue pages from invest.up.gov.in
                            if "invest.up.gov.in" in link:
                                await self.url_queue.put(link)

                progress.update(task_id, description="[green]Crawl complete!")

            await browser.close()

        # Save results
        await self.save_results()

        console.print(f"\n[bold green]Crawl Summary:[/bold green]")
        console.print(f"  Pages visited: {len(self.visited_urls)}")
        console.print(f"  Documents found: {len(self.document_urls)}")

        # Breakdown by type
        type_counts = {}
        for result in self.crawl_results.values():
            type_counts[result.doc_type] = type_counts.get(result.doc_type, 0) + 1
        for doc_type, count in sorted(type_counts.items()):
            console.print(f"    {doc_type.upper()}: {count}")

        return self.crawl_results

    async def save_results(self):
        """Save crawl results to JSON"""
        results_file = LOGS_DIR / "crawl_results.json"
        visited_file = LOGS_DIR / "visited_urls.json"

        # Save crawl results
        results_data = {url: asdict(result) for url, result in self.crawl_results.items()}
        async with aiofiles.open(results_file, "w") as f:
            await f.write(json.dumps(results_data, indent=2, ensure_ascii=False))

        # Save visited URLs
        async with aiofiles.open(visited_file, "w") as f:
            await f.write(json.dumps(list(self.visited_urls), indent=2))

        console.print(f"[blue]Results saved to {results_file}[/blue]")


async def main():
    crawler = InvestUPCrawler()
    results = await crawler.run_crawler()
    return results


if __name__ == "__main__":
    asyncio.run(main())
